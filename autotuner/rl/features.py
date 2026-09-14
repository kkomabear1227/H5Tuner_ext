"""State features for the two agents.

The paper names the agents' inputs but not their encoding: the stopper gets
"the iteration and the performance", the picker gets "the parameter subset and
the best perf achieved during that iteration".  Turning those into numbers is
where most of the agents' actual behaviour is decided, more so than the network
shape, so the encoding is written out explicitly here.

Two properties are deliberate.

Everything is *relative*.  Nothing carries an absolute bandwidth or a wall-clock
second, only ratios against the untuned baseline and against recent history.
That makes the same trained agent usable on a 100 MB/s filesystem and a
100 GB/s one, and it makes the features sign-agnostic: a minimised wall-clock
and a maximised bandwidth produce the same feature vector shape, because gains
are normalised into "better is positive" before they get here.

Per-parameter features are computed by one shared function.  The picker applies
one small network to each parameter's features in turn, so the network never
learns a fixed slot per parameter and the same weights work for a space of 7
parameters or 30.  This is what makes option B scale, and it is also what would
let a trained picker transfer between parameter spaces.
"""

# Lookback used for the recent-improvement features.  Matches the paper's
# 5-iteration reward delay so the state carries the same horizon the reward does.
LOOKBACK = 5

# Relative features are clipped to this range.  A single wild measurement --
# a timeout penalty, say -- would otherwise dominate the input scale.
CLIP = 5.0


def _clip(value):
    return max(-CLIP, min(CLIP, value))


def relative_gain(value, baseline, sense):
    """Improvement over baseline, positive when better, scaled by baseline."""
    if value is None or baseline is None or baseline == 0:
        return 0.0
    gain = (value - baseline) if sense == 'max' else (baseline - value)
    return _clip(gain / abs(baseline))


def _improvement(history, lag, sense):
    """Relative improvement of the best score over the last `lag` iterations."""
    if len(history) <= lag:
        return 0.0
    current = history[-1]
    earlier = history[-(lag + 1)]
    if current is None or earlier is None or earlier == 0:
        return 0.0
    delta = (current - earlier) if sense == 'max' else (earlier - current)
    return _clip(delta / abs(earlier))


def _stall_length(history, sense):
    """Iterations since the best score last changed."""
    if len(history) < 2:
        return 0
    stall = 0
    for index in range(len(history) - 1, 0, -1):
        current, earlier = history[index], history[index - 1]
        if current is None or earlier is None:
            break
        improved = (current > earlier) if sense == 'max' else (current < earlier)
        if improved:
            break
        stall += 1
    return stall


def stopper_state(iteration, horizon, baseline, best_history, sense):
    """Feature vector for the early stopper.  Length 4 + LOOKBACK.

        0   progress through the generation budget
        1   total relative gain achieved so far
        2   relative improvement over the last iteration
        ..  relative improvement over the last 2..LOOKBACK iterations
        -2  length of the current stall, as a fraction of the budget
        -1  constant 1, so the model always has a bias input
    """
    horizon = max(1, horizon)
    best = best_history[-1] if best_history else None
    features = [
        min(1.0, iteration / float(horizon)),
        relative_gain(best, baseline, sense),
    ]
    for lag in range(1, LOOKBACK + 1):
        features.append(_improvement(best_history, lag, sense))
    features.append(min(1.0, _stall_length(best_history, sense)
                        / float(horizon)))
    features.append(1.0)
    return features


STOPPER_INPUTS = 4 + LOOKBACK


class ParameterStats:
    """Per-parameter history the picker conditions on.

    Tracks how often each parameter has been tuned, when it was last tuned, and
    a running estimate of how much tuning it has helped.  The running estimate
    is the picker's own experience; `prior` is what the offline sweep believed
    before the campaign started.
    """

    def __init__(self, names, prior=None, decay=0.7):
        self.names = list(names)
        self.count = {name: 0 for name in self.names}
        self.last_seen = {name: -1 for name in self.names}
        self.effect = {name: 0.0 for name in self.names}
        self.decay = decay
        self.prior = self._normalise(prior or {})
        self.last_subset = set()

    @staticmethod
    def _normalise(scores):
        """Scale scores into [0, 1] so the prior is comparable across traces."""
        if not scores:
            return {}
        values = [value for value in scores.values() if value is not None]
        if not values:
            return {}
        low, high = min(values), max(values)
        if high <= low:
            return {name: 0.5 for name in scores}
        return {name: (value - low) / (high - low)
                for name, value in scores.items()}

    def observe(self, iteration, subset, reward):
        """Fold one iteration's outcome into the per-parameter history."""
        chosen = set(subset or self.names)
        for name in chosen:
            self.count[name] = self.count.get(name, 0) + 1
            self.last_seen[name] = iteration
            previous = self.effect.get(name, 0.0)
            self.effect[name] = (self.decay * previous
                                 + (1.0 - self.decay) * reward)
        self.last_subset = chosen

    def features(self, name, iteration, horizon):
        """Feature vector for one parameter.  Length 5.

            0   was it in the previous subset
            1   how often it has been tuned, as a fraction of iterations
            2   how long since it was last tuned, as a fraction of the budget
            3   running estimate of its effect, from this campaign
            4   prior estimate of its effect, from the offline sweep
        """
        horizon = max(1, horizon)
        elapsed = max(1, iteration + 1)
        last = self.last_seen.get(name, -1)
        since = horizon if last < 0 else min(horizon, iteration - last)
        return [
            1.0 if name in self.last_subset else 0.0,
            min(1.0, self.count.get(name, 0) / float(elapsed)),
            since / float(horizon),
            _clip(self.effect.get(name, 0.0)),
            self.prior.get(name, 0.5),
        ]


PARAM_FEATURES = 5


def picker_global(iteration, horizon, baseline, best_history, sense):
    """Context shared by every parameter this iteration.  Length 5.

        0   progress through the generation budget
        1   total relative gain so far
        2   relative improvement over the last iteration
        3   relative improvement over the last LOOKBACK iterations
        4   length of the current stall, as a fraction of the budget
    """
    horizon = max(1, horizon)
    best = best_history[-1] if best_history else None
    return [
        min(1.0, iteration / float(horizon)),
        relative_gain(best, baseline, sense),
        _improvement(best_history, 1, sense),
        _improvement(best_history, LOOKBACK, sense),
        min(1.0, _stall_length(best_history, sense) / float(horizon)),
    ]


PICKER_GLOBAL = 5
PICKER_INPUTS = PICKER_GLOBAL + PARAM_FEATURES + 1     # + bias


def picker_input(global_features, parameter_features):
    return list(global_features) + list(parameter_features) + [1.0]
