"""TunIO's RL subset picker.

What it decides: which parameters the genetic algorithm is allowed to touch this
generation.

Why it is not a 4,096-way choice
--------------------------------
A subset of twelve parameters is one of 2^12 possibilities, and a campaign of 40
generations offers 40 chances to learn.  Learning 4,096 action values from 40
samples is not possible, so the paper cannot have done that literally, and it
does not say what it did instead.

This implementation scores each parameter separately and takes the top k.  The
network sees one parameter's features at a time and returns one number, so there
are n things to learn rather than 2^n.  Three reasons this reading was chosen
over the alternatives (06-진행-상황.md 4.1):

  * It matches what III-C actually says the picker returns: a subset, ordered by
    descending impact.  A per-parameter score is exactly an impact estimate, and
    sorting it is the ordering.
  * It connects to the paper's own offline pretraining.  III-C pretrains by
    sweeping representative kernels and extracting which parameters matter.
    That produces per-parameter impact -- the same quantity this network
    outputs -- so the pretraining has somewhere to go.  Under a flat or
    factorised encoding the mapping from sweep results to initial policy would
    be another unspecified choice.
  * It isolates what the RL contributes.  The static selector already ranks
    parameters and takes the top k; switching to this one changes only where the
    ranking comes from.  An ablation between them measures the learning, not a
    change of mechanism.

The cost is that subset *size* is not learned -- k is a setting.  The paper does
not specify a size or a schedule either way.

Why a bandit rather than multi-step Q-learning
----------------------------------------------
The score for a parameter is trained as a one-step value: after an iteration,
every parameter that was tuned gets its score pulled toward the reward that
iteration produced.  That is a contextual bandit, which is also what III-C calls
the State Observer, so this keeps both of the paper's named pieces coherent
instead of picking one and ignoring the other.

Reward
------
III-C gives the shape:

    reward = norm_perf(perf) - norm_param(number of parameters in the subset)

with a 5-iteration delay.  Two substitutions were needed.

`norm_perf` divides by `1 / (BW_single * num_nodes)` in the paper, and
`BW_single` is never defined.  We use the gain over the untuned baseline,
divided by the baseline -- dimensionless, and available without an extra
calibration run.

`norm_param` divides by the total parameter count, which is unambiguous.  The
relative weight of the two terms is not given; `size_penalty` defaults to 1.0,
matching the paper's plain difference.

The delay is honoured literally: an iteration's subset is not scored until five
iterations later, using the improvement realised over that window.  Without it a
subset that pays off slowly would be punished for the plateau it has to cross.
"""

import math
import random

from . import features as feature_module
from . import nets
from ..early_stopping import REWARD_DELAY
from ..subset import SubsetSelector

DEFAULT_SIZE_PENALTY = 1.0
DEFAULT_EPSILON = 0.3
DEFAULT_EPSILON_FLOOR = 0.05
DEFAULT_WARM_ROUNDS = 200


class BanditPicker(SubsetSelector):
    """Scores each parameter, tunes the top k.

    space           the parameter space
    objective       supplies `sense`
    horizon         generation budget, used to normalise progress features
    size            parameters tuned per iteration; defaults to a third of the
                    free ones, rounded up
    prior           optional {parameter name: impact score} from an offline
                    sweep.  Enters both as an input feature and as the
                    warm-start target, which is how the paper's pretraining
                    reaches the policy.
    size_penalty    weight on the subset-size term of the reward
    epsilon         exploration rate, decayed toward `epsilon_floor`
    """

    name = 'rl'

    def __init__(self, space, objective, horizon, size=None, prior=None,
                 size_penalty=DEFAULT_SIZE_PENALTY, epsilon=DEFAULT_EPSILON,
                 epsilon_floor=DEFAULT_EPSILON_FLOOR, seed=None,
                 warm_rounds=DEFAULT_WARM_ROUNDS, model=None):
        self.space = space
        self.objective = objective
        self.horizon = max(1, horizon)
        self.names = list(space.free_names)
        self.size = size if size else max(1, math.ceil(len(self.names) / 3))
        self.size_penalty = size_penalty
        self.epsilon = epsilon
        self.epsilon_floor = epsilon_floor
        self.rng = random.Random(seed)
        self.model = model or nets.build(feature_module.PICKER_INPUTS, 1,
                                         seed=seed)
        self.stats = feature_module.ParameterStats(self.names, prior=prior)

        self.baseline = None
        self.best_history = []
        self._pending = []
        self.warm_report = None
        if prior and warm_rounds:
            self.warm_report = self.warm_start(prior, rounds=warm_rounds)

    # -- scoring ------------------------------------------------------------

    def _inputs(self, iteration, global_features):
        return {
            name: feature_module.picker_input(
                global_features,
                self.stats.features(name, iteration, self.horizon))
            for name in self.names}

    def scores(self, iteration, campaign):
        """Estimated impact of each parameter right now, highest first."""
        global_features = feature_module.picker_global(
            iteration=iteration, horizon=self.horizon,
            baseline=self.baseline, best_history=self.best_history,
            sense=self.objective.sense)
        inputs = self._inputs(iteration, global_features)
        scored = [(name, self.model.predict(vector)[0])
                  for name, vector in inputs.items()]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored, inputs

    # -- SubsetSelector -----------------------------------------------------

    def select(self, iteration, campaign):
        if self.baseline is None:
            self.baseline = campaign.baseline

        scored, inputs = self.scores(iteration, campaign)
        chosen = [name for name, _ in scored[:self.size]]

        # Exploration.  Parameters that are never selected never generate
        # feedback, so without this the initial ordering would be permanent.
        rate = max(self.epsilon_floor,
                   self.epsilon * (1.0 - iteration / float(self.horizon)))
        if self.rng.random() < rate and len(self.names) > self.size:
            outsiders = [name for name, _ in scored[self.size:]]
            chosen[self.rng.randrange(len(chosen))] = self.rng.choice(outsiders)

        subset = set(chosen)
        self._pending.append({
            'iteration': iteration,
            'subset': subset,
            'inputs': {name: inputs[name] for name in subset},
        })
        return subset

    def observe(self, iteration, subset, campaign):
        self.baseline = campaign.baseline
        self.best_history.append(campaign.best_score)
        self._settle(iteration)

    def finish(self, campaign):
        """Score whatever is still pending, at the end of the campaign.

        Called when the run stops before the delayed rewards have matured.  The
        window is shorter than REWARD_DELAY, so these samples are noisier, but
        discarding them would throw away the last five iterations entirely.
        """
        self._settle(len(self.best_history) - 1, force=True)

    # -- learning -----------------------------------------------------------

    def _reward(self, entry, now):
        """Delayed reward for one past iteration's subset."""
        start = entry['iteration']
        end = min(now, start + REWARD_DELAY)
        if end <= start or start >= len(self.best_history):
            return None
        end = min(end, len(self.best_history) - 1)
        if end <= start:
            return None
        earlier = self.best_history[start]
        later = self.best_history[end]
        if earlier is None or later is None or earlier == 0:
            return None
        delta = ((later - earlier) if self.objective.sense == 'max'
                 else (earlier - later))
        gain = delta / abs(earlier)
        penalty = self.size_penalty * len(entry['subset']) / float(
            max(1, len(self.names)))
        return gain - penalty

    def _settle(self, now, force=False):
        matured = []
        remaining = []
        for entry in self._pending:
            ready = force or (now - entry['iteration']) >= REWARD_DELAY
            if ready:
                matured.append(entry)
            else:
                remaining.append(entry)
        self._pending = remaining

        batch = []
        for entry in matured:
            reward = self._reward(entry, now)
            if reward is None:
                continue
            self.stats.observe(entry['iteration'], entry['subset'], reward)
            for vector in entry['inputs'].values():
                batch.append((vector, [reward], [1]))
        if batch:
            self.rng.shuffle(batch)
            self.model.fit(batch)

    # -- offline pretraining ------------------------------------------------

    def warm_start(self, prior, rounds=DEFAULT_WARM_ROUNDS):
        """Fit the scorer so its initial ranking reproduces `prior`.

        This is where the paper's offline pretraining lands.  III-C sweeps
        representative kernels and extracts which parameters matter, but never
        says how that becomes an initial policy.  Regressing the scorer onto the
        swept impact does it directly: before the campaign has any experience,
        the greedy subset is the sweep's top k, and every iteration afterwards
        moves the scores by what actually happened.

        The prior is also an input feature, so the network can learn to
        discount it if the campaign disagrees.
        """
        normalised = feature_module.ParameterStats._normalise(prior)  # noqa: SLF001
        if not normalised:
            return None

        rng = random.Random(self.rng.random())
        losses = []
        for _ in range(rounds):
            batch = []
            # Sample synthetic contexts so the warm start holds across the whole
            # campaign, not only at iteration zero.
            iteration = rng.randrange(self.horizon)
            global_features = feature_module.picker_global(
                iteration=iteration, horizon=self.horizon, baseline=1.0,
                best_history=[1.0 + rng.random()], sense='max')
            for name in self.names:
                target = normalised.get(name, 0.5)
                vector = feature_module.picker_input(
                    global_features,
                    self.stats.features(name, iteration, self.horizon))
                batch.append((vector, [target], [1]))
            rng.shuffle(batch)
            losses.append(self.model.fit(batch))

        ranked = sorted(normalised.items(), key=lambda pair: pair[1],
                        reverse=True)
        return {
            'backend': self.model.backend,
            'rounds': rounds,
            'final_loss': losses[-1] if losses else None,
            'prior_top': [name for name, _ in ranked[:self.size]],
        }
