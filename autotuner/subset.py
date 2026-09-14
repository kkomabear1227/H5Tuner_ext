"""Which parameters to tune this iteration.

TunIO's Smart Configuration Generation picks a subset of the parameters each
iteration and orders it by descending impact, so that the search space the
genetic algorithm actually explores stays small and the high-payoff knobs move
first.  IV-B reports the effect: FLASH reaches 2.3 GB/s at iteration 6 instead
of 43.

Three selectors:

    AllParameters   tune everything, every iteration.  H5Tuner's behaviour.
    StaticRanking   a fixed impact order, computed offline from a trace.  This
                    is the non-RL half of the paper's own pretraining, used
                    directly.  Scales linearly in the number of parameters.
    build_rl_picker TunIO's contribution.  A contextual bandit that scores each
                    parameter and takes the top k -- see rl/picker.py.

A caution about StaticRanking.  The paper's V-C criticises an earlier H5Tuner
follow-up for reducing to a fixed set of impactful parameters, on the grounds
that the choice does not generalise across applications or scales.  Using the
static ranking alone rebuilds the thing the paper argues against.  It is a
baseline and an ablation arm, not the destination.
"""

import math


class SubsetSelector:
    """Base class.  `select` returns a set of parameter names, or None for all."""

    name = None

    def select(self, iteration, campaign):
        raise NotImplementedError

    def observe(self, iteration, subset, campaign):
        """Feedback after the iteration.  Stateless selectors ignore it."""

    def finish(self, campaign):
        """Campaign is over.  Learners may settle pending feedback."""


class AllParameters(SubsetSelector):
    """No subsetting.  What H5Tuner does."""

    name = 'all'

    def select(self, iteration, campaign):
        return None


class StaticRanking(SubsetSelector):
    """Tune the top-k parameters of a fixed impact ranking.

    ranking     parameter names, most impactful first
    size        how many to tune per iteration.  Defaults to a third of the
                free parameters, rounded up.
    grow_every  when set, start from `size` and add one parameter every this
                many iterations.  The paper does not specify a schedule -- it
                says only that a subset is selected and ordered -- so both the
                fixed and growing schedules are ours.
    """

    name = 'static'

    def __init__(self, space, ranking, size=None, grow_every=None):
        free = set(space.free_names)
        self.ranking = [name for name in ranking if name in free]
        missing = [name for name in space.free_names
                   if name not in set(self.ranking)]
        # Unranked parameters go last rather than being dropped, so a partial
        # ranking degrades to "tune the ranked ones first" instead of silently
        # excluding knobs.
        self.ranking.extend(missing)
        self.size = size if size else max(1, math.ceil(len(self.ranking) / 3))
        self.grow_every = grow_every

    def select(self, iteration, campaign):
        count = self.size
        if self.grow_every:
            count += iteration // self.grow_every
        count = max(1, min(count, len(self.ranking)))
        return set(self.ranking[:count])


def build_rl_picker(space, objective, horizon, **kwargs):
    """TunIO's RL subset picker.

    Implemented in rl/picker.py as a contextual bandit that scores each
    parameter and takes the top k.  Imported lazily so that a run using
    --subset all or --subset static never touches the RL code path.

    Why per-parameter scoring rather than a choice among all 2^n subsets, and
    what that costs, is argued in rl/picker.py and 06-진행-상황.md 4.1.
    """
    from .rl.picker import BanditPicker
    return BanditPicker(space=space, objective=objective, horizon=horizon,
                        **kwargs)


# ---------------------------------------------------------------------------
# Offline impact ranking
# ---------------------------------------------------------------------------

def rank_from_trace(space, trace, objective):
    """Rank parameters by how much varying them moves the objective.

    For each parameter, group the recorded measurements by that parameter's
    value, take each group's mean score, and use the spread of those group
    means as the effect size.  A parameter whose value does not change the mean
    score ranks last.

    This is NOT the paper's method.  III-C says a PCA is performed on the swept
    parameters with respect to perf, but does not say how component loadings
    become a per-parameter ranking, and PCA of inputs alone carries no
    information about the target.  A one-way effect size answers the question
    the ranking is actually for -- which knob moves the objective -- with no
    dependencies and no interpretation gap.  Implementing something closer to
    the paper is a TODO.

    Interaction effects are invisible to this method.  Two parameters that only
    matter together will both look inert.
    """
    groups = {name: {} for name in space.free_names}
    for entry in trace._entries.values():          # noqa: SLF001 - same package
        for name in groups:
            if name not in entry.values:
                continue
            groups[name].setdefault(entry.values[name], []).append(entry.score)

    effects = []
    for name, buckets in groups.items():
        means = [sum(scores) / float(len(scores))
                 for scores in buckets.values() if scores]
        if len(means) < 2:
            effects.append((name, 0.0, len(buckets)))
            continue
        centre = sum(means) / float(len(means))
        variance = sum((value - centre) ** 2 for value in means) / len(means)
        effects.append((name, math.sqrt(variance), len(buckets)))

    effects.sort(key=lambda row: row[1], reverse=True)
    return effects


def format_ranking(effects, objective):
    lines = ['{0:<24}{1:>14}  {2:>7}'.format('parameter',
                                             'effect ({0})'.format(
                                                 objective.unit or '-'),
                                             'levels')]
    for name, effect, levels in effects:
        lines.append('{0:<24}{1:>14.3f}  {2:>7}'.format(name, effect, levels))
    return '\n'.join(lines)


SUBSET_SELECTORS = ('all', 'static', 'rl')
