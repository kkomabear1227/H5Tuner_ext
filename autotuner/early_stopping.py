"""When to stop tuning.

Three stoppers, matching the three arms the paper compares in IV-C.

    NeverStop           run the full generation budget.  "H5Tuner with No Stop".
    HeuristicStopper    stop after `window` iterations without a `threshold`
                        relative improvement.  The paper's baseline, at 5% over
                        5 iterations, chosen because 5% is a common measure of
                        statistical significance.
    RLStopper           TunIO's contribution.  Trained offline on synthetic
                        logarithmic tuning curves.

The heuristic's failure mode is the whole reason TunIO's version exists.  In
Fig 10(a) the HACC curve plateaus between iterations 10 and 20; the heuristic
reads that as convergence and quits at iteration 14 with 1.2 GB/s, while
continuing would have reached 2.2 GB/s.  That is 83% of the available gain
thrown away.  Reproducing that plateau on a replayed trace is the first thing to
check before claiming the RL stopper helps.

The synthetic curve generator below is specified well enough in III-D to
reimplement: performance follows a logarithmic curve, each simulated
application gets its own initial value and growth rate, and noise takes the form
of random downward shifts to model briefly picking a bad parameter subset.
"""

import math

# Paper's heuristic thresholds (IV-C).
HEURISTIC_THRESHOLD = 0.05
HEURISTIC_WINDOW = 5
# Reward delay used by both RL components (III-C, III-D).
REWARD_DELAY = 5


class Stopper:
    """Base class.

    `should_stop` is called once per iteration with the campaign's history and
    returns (stop, reason).  Returning a reason even when not stopping is
    allowed; it is only reported when stop is True.
    """

    name = None

    def reset(self):
        pass

    def should_stop(self, iteration, best_history):
        """best_history is the best score after each iteration so far."""
        raise NotImplementedError


class NeverStop(Stopper):
    """Exhaust the generation budget."""

    name = 'never'

    def should_stop(self, iteration, best_history):
        return False, ''


class HeuristicStopper(Stopper):
    """Stop when relative improvement stalls.

    Improvement is measured in the objective's own direction, so this works for
    both a minimised wall-clock and a maximised bandwidth.
    """

    name = 'heuristic'

    def __init__(self, objective, threshold=HEURISTIC_THRESHOLD,
                 window=HEURISTIC_WINDOW):
        self.objective = objective
        self.threshold = threshold
        self.window = window

    def should_stop(self, iteration, best_history):
        if len(best_history) <= self.window:
            return False, ''
        current = best_history[-1]
        reference = best_history[-(self.window + 1)]
        if reference in (0, None) or current is None:
            return False, ''
        if self.objective.sense == 'max':
            improvement = (current - reference) / abs(reference)
        else:
            improvement = (reference - current) / abs(reference)
        if improvement < self.threshold:
            return True, ('no {0:.0%} improvement over {1} iterations '
                          '(saw {2:.2%})'.format(self.threshold, self.window,
                                                 improvement))
        return False, ''


class BudgetStopper(Stopper):
    """Stop once a wall-clock tuning budget is spent.

    Not part of either paper, but every real campaign has an allocation, and
    Fig 12's viability analysis is expressed in exactly these terms.
    """

    name = 'budget'

    def __init__(self, max_minutes, campaign):
        self.max_minutes = max_minutes
        self.campaign = campaign

    def should_stop(self, iteration, best_history):
        spent = self.campaign.tuning_seconds / 60.0
        if spent >= self.max_minutes:
            return True, 'tuning budget of {0:.0f} min exhausted'.format(
                self.max_minutes)
        return False, ''


class CompositeStopper(Stopper):
    """Stop as soon as any member wants to."""

    name = 'composite'

    def __init__(self, members):
        self.members = list(members)

    def reset(self):
        for member in self.members:
            member.reset()

    def should_stop(self, iteration, best_history):
        for member in self.members:
            stop, reason = member.should_stop(iteration, best_history)
            if stop:
                return True, '{0}: {1}'.format(member.name, reason)
        return False, ''


# ---------------------------------------------------------------------------
# Synthetic tuning curves, for offline training
# ---------------------------------------------------------------------------

class LogCurve:
    """One simulated application's tuning curve.

    Performance rises logarithmically and flattens:

        perf(i) = initial + scale * log(1 + growth * i)

    Noise is a random downward shift applied with probability `noise_rate`,
    which is III-D's model for "the wrong parameter was chosen briefly before
    adjusting".  The shift persists for one iteration only; the underlying
    curve is unaffected, exactly as a bad subset choice does not undo the
    elitist best.
    """

    def __init__(self, initial, scale, growth, noise_rate=0.15,
                 noise_depth=0.25, plateau_at=None, plateau_length=0):
        self.initial = initial
        self.scale = scale
        self.growth = growth
        self.noise_rate = noise_rate
        self.noise_depth = noise_depth
        # An optional flat stretch, so the generator can produce the trap the
        # heuristic stopper falls into (Fig 10(a), iterations 10-20).
        self.plateau_at = plateau_at
        self.plateau_length = plateau_length

    def true_value(self, iteration):
        effective = iteration
        if self.plateau_at is not None and iteration > self.plateau_at:
            effective = min(iteration,
                            self.plateau_at) + max(
                                0, iteration - self.plateau_at
                                - self.plateau_length)
        return self.initial + self.scale * math.log1p(self.growth * effective)

    def observed(self, iteration, rng):
        value = self.true_value(iteration)
        if rng.random() < self.noise_rate:
            value *= (1.0 - self.noise_depth * rng.random())
        return value

    def best_achievable(self, horizon):
        return self.true_value(horizon)


    @classmethod
    def with_total_gain(cls, initial, total_gain_ratio, growth, horizon,
                       **kwargs):
        """Build a curve that gains `total_gain_ratio` x initial by `horizon`.

        Parameterising by the total gain rather than by the raw log scale keeps
        the generated population realistic and comparable.  A curve's steepness
        is otherwise arbitrary, and if it is too steep the agent never sees an
        iteration whose improvement is not worth the cost, so it learns to run
        forever.

        Reference points from the paper: HACC goes from 0.55 to 2.2 GB/s under
        tuning (IV-C), a total gain of 3x the untuned score; the improvements in
        Fig 2 are of the same order.  Ratios are sampled around that.
        """
        span = math.log1p(growth * max(1, horizon))
        scale = initial * total_gain_ratio / span if span > 0 else initial
        return cls(initial=initial, scale=scale, growth=growth, **kwargs)


def sample_curves(rng, count, horizon=50, plateau_fraction=0.4):
    """A training population of simulated applications.

    `plateau_fraction` of them get a flat stretch, so an agent trained on this
    population has to learn to sit through one rather than treating any
    stall as convergence.  That flat stretch is the Fig 10(a) trap.
    """
    curves = []
    for _ in range(count):
        plateau_at = None
        plateau_length = 0
        if rng.random() < plateau_fraction:
            plateau_at = rng.randint(int(horizon * 0.1), int(horizon * 0.5))
            plateau_length = rng.randint(int(horizon * 0.1),
                                         int(horizon * 0.3))
        curves.append(LogCurve.with_total_gain(
            initial=rng.uniform(0.3, 1.2),
            # Total improvement over the whole budget, as a multiple of the
            # untuned score.  Spans "tuning barely helped" to "tuning
            # quadrupled it", bracketing the paper's reported gains.
            total_gain_ratio=rng.uniform(0.4, 4.0),
            growth=rng.uniform(0.1, 1.5),
            horizon=horizon,
            noise_rate=rng.uniform(0.05, 0.25),
            noise_depth=rng.uniform(0.1, 0.4),
            plateau_at=plateau_at,
            plateau_length=plateau_length))
    return curves


# ---------------------------------------------------------------------------
# RL stopper
# ---------------------------------------------------------------------------

def build_rl_stopper(objective, horizon, **kwargs):
    """TunIO's RL early stopper.

    Implemented in rl/stopper.py as a DQN over the engineered state in
    rl/features.py, trained offline on the synthetic curves above.  Imported
    lazily so a heuristic run never pays for it.

    Why one network end to end rather than the paper's separate State Observer
    and Action Decider is argued in rl/stopper.py and
    06-진행-상황.md 4.2.
    """
    from .rl.stopper import DQNStopper
    return DQNStopper(objective=objective, horizon=horizon, **kwargs)


STOPPERS = ('never', 'heuristic', 'rl', 'budget')


def build_stopper(name, objective, campaign=None, max_minutes=None,
                  horizon=40, rl_kwargs=None):
    if name == 'never':
        stopper = NeverStop()
    elif name == 'heuristic':
        stopper = HeuristicStopper(objective)
    elif name == 'rl':
        stopper = build_rl_stopper(objective, horizon, **(rl_kwargs or {}))
    elif name == 'budget':
        if max_minutes is None:
            raise ValueError('--stopper budget requires --max-minutes')
        stopper = BudgetStopper(max_minutes, campaign)
    else:
        raise ValueError('unknown stopper {0!r}; choose from {1}'.format(
            name, ', '.join(STOPPERS)))

    if max_minutes is not None and name != 'budget':
        return CompositeStopper([stopper, BudgetStopper(max_minutes,
                                                        campaign)])
    return stopper
