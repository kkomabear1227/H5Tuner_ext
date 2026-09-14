"""Campaign bookkeeping and Return on Tuning Investment.

TunIO measures itself with RoTI, defined in IV as

    RoTI(t) = (perf_achieved(t) - perf_achieved(0)) / t

the performance gained divided by the tuning time spent, in MB/s per minute.
It is the metric the paper's central claim rests on: TunIO reaches an RoTI of
215 against H5Tuner-with-heuristic-stop's 41.6, while conceding 3% of peak
bandwidth.  Any comparison against those numbers has to compute it the same way.

Two things to keep straight.

perf_achieved(0) is the *untuned* configuration, so a campaign has to measure
the defaults before it starts searching, and that measurement costs budget like
any other.

RoTI assumes a maximised objective.  Under --objective walltime the gain is
seconds saved rather than bandwidth won, so the number is not comparable to the
paper's.  It is still monotone in the right direction, and `unit` says which one
you are looking at.
"""


class Sample:
    """One evaluation, as the campaign saw it."""

    __slots__ = ('iteration', 'tuning_seconds', 'score', 'best', 'source')

    def __init__(self, iteration, tuning_seconds, score, best, source):
        self.iteration = iteration
        self.tuning_seconds = tuning_seconds
        self.score = score
        self.best = best
        self.source = source


class Generation:
    """One iteration of the search, as the framework reported it."""

    __slots__ = ('index', 'tuning_seconds', 'best', 'mean', 'evaluations',
                 'subset', 'note')

    def __init__(self, index, tuning_seconds, best, mean, evaluations,
                 subset=None, note=''):
        self.index = index
        self.tuning_seconds = tuning_seconds
        self.best = best
        self.mean = mean
        self.evaluations = evaluations
        # Parameter names tuned this iteration, when the framework selects a
        # subset.  None means "all of them".
        self.subset = subset
        self.note = note


class Campaign:
    """Accumulates a run's history and derives its metrics.

    The framework reports every evaluation via `record` and every iteration via
    `close_generation`.  Nothing here decides anything -- it only observes, so
    the same bookkeeping applies to every framework.
    """

    def __init__(self, objective, framework, space):
        self.objective = objective
        self.framework = framework
        self.space = space
        self.baseline = None
        self.baseline_point = None
        self.samples = []
        self.generations = []
        self.best_score = None
        self.best_point = None
        self.stopped_at = None
        self.stop_reason = None

    # -- reporting ----------------------------------------------------------

    def set_baseline(self, result):
        """Record the untuned configuration's score."""
        self.baseline = result.score
        self.baseline_point = result.point

    def record(self, result, tuning_seconds, iteration):
        if self.best_score is None or self.objective.better(result.score,
                                                            self.best_score):
            self.best_score = result.score
            self.best_point = result.point
        self.samples.append(Sample(iteration, tuning_seconds, result.score,
                                   self.best_score, result.source))

    def close_generation(self, index, tuning_seconds, scores, evaluations,
                         subset=None, note=''):
        scores = list(scores)
        self.generations.append(Generation(
            index=index,
            tuning_seconds=tuning_seconds,
            best=self.objective.best_of(scores) if scores else None,
            mean=sum(scores) / float(len(scores)) if scores else None,
            evaluations=evaluations,
            subset=None if subset is None else tuple(subset),
            note=note))

    def stop(self, iteration, reason):
        self.stopped_at = iteration
        self.stop_reason = reason

    # -- derived ------------------------------------------------------------

    @property
    def unit(self):
        return '{0} per tuning minute'.format(self.objective.unit)

    def gain(self, score):
        """Improvement over the baseline, positive when better.

        Sign is normalised so that a bigger gain is always better, whichever
        way the objective points.
        """
        if self.baseline is None or score is None:
            return None
        if self.objective.sense == 'max':
            return score - self.baseline
        return self.baseline - score

    def roti(self, score, tuning_seconds):
        gain = self.gain(score)
        if gain is None or tuning_seconds <= 0:
            return None
        return gain / (tuning_seconds / 60.0)

    def roti_curve(self):
        """[(tuning minutes, RoTI), ...] over the campaign."""
        curve = []
        for sample in self.samples:
            value = self.roti(sample.best, sample.tuning_seconds)
            if value is not None:
                curve.append((sample.tuning_seconds / 60.0, value))
        return curve

    def peak_roti(self):
        """Best RoTI reached, and when.

        The paper reports peak RoTI rather than final RoTI, because RoTI decays
        once tuning stops paying off -- the denominator keeps growing.
        """
        best = None
        for minutes, value in self.roti_curve():
            if best is None or value > best[1]:
                best = (minutes, value)
        return best

    def final_roti(self):
        if not self.samples:
            return None
        last = self.samples[-1]
        return self.roti(last.best, last.tuning_seconds)

    @property
    def tuning_seconds(self):
        return self.samples[-1].tuning_seconds if self.samples else 0.0

    def summary(self):
        peak = self.peak_roti()
        return {
            'framework': self.framework,
            'space': self.space.name,
            'objective': self.objective.name,
            'sense': self.objective.sense,
            'baseline': self.baseline,
            'best': self.best_score,
            'best_config': (None if self.best_point is None
                            else self.space.decode(self.best_point)),
            'gain': self.gain(self.best_score),
            'evaluations': len(self.samples),
            'generations': len(self.generations),
            'tuning_minutes': self.tuning_seconds / 60.0,
            'roti_final': self.final_roti(),
            'roti_peak': None if peak is None else peak[1],
            'roti_peak_minutes': None if peak is None else peak[0],
            'stopped_at': self.stopped_at,
            'stop_reason': self.stop_reason,
            # Per-generation history, so the improvement curve survives the
            # run.  Without it the only record of "did performance climb" is
            # stdout, and a job that truncates its log loses the answer -- as
            # happened on the first real Nurion campaign.
            'history': [
                {
                    'generation': g.index,
                    'best': g.best,
                    'mean': g.mean,
                    'running_best': running,
                    'evaluations': g.evaluations,
                    'tuning_minutes': g.tuning_seconds / 60.0,
                    'roti': self.roti(running, g.tuning_seconds),
                    'subset': (None if g.subset is None
                               else sorted(g.subset)),
                }
                for g, running in zip(self.generations,
                                      self._running_bests())
            ],
        }

    def _running_bests(self):
        """Best-so-far after each generation, in order."""
        out = []
        best = None
        for generation in self.generations:
            if best is None or self.objective.better(generation.best, best):
                best = generation.best
            out.append(best)
        return out


def format_summary(campaign):
    """Human-readable campaign summary."""
    data = campaign.summary()
    objective = campaign.objective
    lines = []
    lines.append('framework      {0}'.format(data['framework']))
    lines.append('space          {0} ({1} free parameters, {2:,} configurations)'
                 .format(data['space'], len(campaign.space.free_names),
                         campaign.space.size()))
    lines.append('objective      {0} ({1}imise, {2})'.format(
        data['objective'], data['sense'], objective.unit or 'unitless'))
    lines.append('')

    def show(label, value):
        if value is None:
            lines.append('{0:<15}n/a'.format(label))
        else:
            lines.append('{0:<15}{1:.3f} {2}'.format(label, value,
                                                     objective.unit))

    show('baseline', data['baseline'])
    show('best', data['best'])
    show('gain', data['gain'])
    lines.append('')
    lines.append('evaluations    {0}'.format(data['evaluations']))
    lines.append('generations    {0}'.format(data['generations']))
    lines.append('tuning time    {0:.1f} min'.format(data['tuning_minutes']))
    if data['roti_peak'] is not None:
        lines.append('RoTI peak      {0:.3f} {1} at {2:.1f} min'.format(
            data['roti_peak'], campaign.unit, data['roti_peak_minutes']))
    if data['roti_final'] is not None:
        lines.append('RoTI final     {0:.3f} {1}'.format(
            data['roti_final'], campaign.unit))
    if data['stopped_at'] is not None:
        lines.append('stopped        iteration {0} ({1})'.format(
            data['stopped_at'], data['stop_reason']))
    if data['best_config']:
        lines.append('')
        lines.append('best configuration')
        for name in sorted(data['best_config']):
            lines.append('  {0:<24}{1}'.format(name,
                                               data['best_config'][name]))
    return '\n'.join(lines)
