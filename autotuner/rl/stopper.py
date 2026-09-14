"""TunIO's RL early stopper.

What it decides: after each generation, whether to run another one.

How it is trained: entirely offline, on synthetic tuning curves.  III-D says
performance during tuning follows a logarithmic curve (Fig 2), so the paper
generates such curves, adds downward noise to model briefly picking a bad
parameter subset, varies the initial value and growth rate per simulated
application, and trains until the average reward stops improving.  All of that
is specified and lives in early_stopping.py.

Reward
------
III-D states the goal but not the function: "stop when there is no measurable
improvement in performance over recent tuning iterations ... balance the
performance gained with the time spent tuning."  So the reward is

    continuing into iteration i+1 earns
        (relative improvement of the best score) - step_cost

    stopping earns nothing further and ends the episode

`step_cost` is what one generation of evaluations is deemed to be worth.  The
learned rule then reads:

    continue while the *expected* relative improvement per iteration over the
    next five iterations exceeds step_cost

**Not** RoTI.  Maximising RoTI directly would be wrong, and it is worth saying
why, because it looks like the obvious choice.  RoTI is a gain divided by
cumulative cost.  The gain saturates logarithmically while the cost grows
linearly, so RoTI peaks within the first few iterations and declines forever
after.  An agent maximising it stops almost immediately.  The paper's own
numbers show its stopper does not behave that way: IV-C has it stopping at
generation 35 of 50 and reaching 90.5% of the best available RoTI, not 100%.
RoTI is how the paper *reports* results, not what its agent optimises.

`step_cost` defaults to HEURISTIC_THRESHOLD / HEURISTIC_WINDOW, that is
0.05 / 5 = 0.01.  This is deliberate calibration, not a tuned constant.  It puts
the RL stopper on exactly the cost/benefit threshold the paper's heuristic
baseline uses -- 5% improvement over 5 iterations is 1% per iteration -- so an
ablation between the two isolates the one thing that actually differs:

    the heuristic looks *backward* at improvement already realised
    the agent looks *forward* at improvement it predicts

A plateau the curve eventually escapes is indistinguishable in hindsight and
distinguishable in prospect.  That is the Fig 10(a) failure, where the heuristic
quits at iteration 14 and forfeits 83% of the available gain.

The paper's "5-iteration delay on the reward function" is implemented as 5-step
returns in the Q target.  That credits an action with what happened over the
following five iterations without double-counting overlapping windows.

Architecture
------------
One network, trained end to end: the hidden layers stand in for the paper's
State Observer and the output layer for its Action Decider.  This is a standard
DQN.  The paper describes the two as separate components but specifies neither,
and at this size -- a state of 8 numbers, two actions -- the split does not
change the policy.  See 06-진행-상황.md 4.2.

Because the synthetic environment is policy-independent (the curve evolves the
same way whatever the agent does), training uses fitted Q iteration over every
decision point of every curve rather than sampled rollouts.  There is no
exploration problem to solve, and every curve contributes every transition.
"""

import random

from . import features as feature_module
from . import nets
from ..early_stopping import (HEURISTIC_THRESHOLD, HEURISTIC_WINDOW,
                              REWARD_DELAY, Stopper, sample_curves)

CONTINUE, STOP = 0, 1

# What one generation of evaluations is deemed to be worth, in units of relative
# improvement over the untuned baseline.  Calibrated to the heuristic baseline's
# threshold so the two stoppers differ only in hindsight versus foresight.
DEFAULT_STEP_COST = HEURISTIC_THRESHOLD / HEURISTIC_WINDOW

DEFAULT_CURVES = 240
DEFAULT_HORIZON = 40
DEFAULT_ROUNDS = 80
DEFAULT_EPOCHS = 40
DEFAULT_BATCH = 256
DEFAULT_DISCOUNT = 0.98
# Rounds to run before the stagnation check may fire.  Without a floor the check
# triggers during warm-up, while the policy is still uniformly always-continue
# and its score is flat for reasons that have nothing to do with convergence.
MIN_ROUNDS = 35
# III-D: train until the average reward stagnates, "5% or less increase across
# five iterations".
STAGNATION_THRESHOLD = 0.05
STAGNATION_WINDOW = 5


def _curve_history(curve, horizon, rng):
    """Best-so-far series for one simulated application.

    The observed value can dip below the running best because of the noise
    model, but "best so far" is monotone -- elitism carries the best
    configuration forward, so a bad iteration never loses ground.
    """
    history = []
    best = None
    for iteration in range(horizon):
        observed = curve.observed(iteration, rng)
        best = observed if best is None else max(best, observed)
        history.append(best)
    return history


class Episode:
    """One simulated tuning run, precomputed.

    Holds the best-so-far history, the per-step rewards and the state at every
    decision point.  None of it depends on the agent's actions, so an episode is
    built once and reused across every training round.
    """

    def __init__(self, curve, horizon, rng, step_cost):
        self.horizon = horizon
        self.history = _curve_history(curve, horizon, rng)
        self.baseline = curve.true_value(0)
        self.step_cost = step_cost

        # rewards[i] is earned by continuing from decision point i into i + 1.
        #
        # Improvement is measured against the CURRENT best, not the baseline.
        # That is the same reference the heuristic stopper uses -- it compares
        # against the best five iterations ago -- and it is what makes
        # `step_cost` mean the same thing to both.  Measuring against the
        # baseline instead would make a late iteration's 1% look like 3% once
        # the best has tripled, and the agent would never stop.
        self.rewards = []
        for index in range(horizon):
            if index + 1 >= horizon:
                self.rewards.append(0.0)
                continue
            current = self.history[index]
            scale = abs(current) if current else 1.0
            gained = (self.history[index + 1] - current) / scale
            self.rewards.append(gained - step_cost)

        self.states = [
            feature_module.stopper_state(
                iteration=index, horizon=horizon, baseline=self.baseline,
                best_history=self.history[:index + 1], sense='max')
            for index in range(horizon)]

    def return_from(self, start, stop_at):
        """Undiscounted return for continuing from `start` until `stop_at`."""
        return sum(self.rewards[start:stop_at])

    def best_return(self):
        """Return of the best possible stopping time, for scoring a policy.

        Because rewards can be negative, the best stopping time is the prefix
        with the largest sum, which is a running maximum over prefix sums.
        """
        best = 0.0
        running = 0.0
        for reward in self.rewards:
            running += reward
            best = max(best, running)
        return best


class DQNStopper(Stopper):
    """Trained early stopper.

    Construct with `train=True` (the default) to fit on synthetic curves
    immediately.  Training is offline and needs neither the application nor a
    trace, so it happens at startup.
    """

    name = 'rl'

    def __init__(self, objective, horizon=DEFAULT_HORIZON, model=None,
                 seed=None, train=True, curves=DEFAULT_CURVES,
                 rounds=DEFAULT_ROUNDS, discount=DEFAULT_DISCOUNT,
                 step_cost=DEFAULT_STEP_COST, verbose=False):
        self.objective = objective
        self.horizon = horizon
        self.discount = discount
        self.step_cost = step_cost
        self.seed = seed
        self.model = model or nets.build(feature_module.STOPPER_INPUTS, 2,
                                         seed=seed)
        self.baseline = None
        self.trained = False
        self.training_report = None
        if train:
            self.training_report = self.fit(curves=curves, rounds=rounds,
                                           verbose=verbose)

    # -- inference ----------------------------------------------------------

    def bind_baseline(self, baseline):
        """Tell the stopper the untuned score, so gains can be normalised."""
        self.baseline = baseline

    def should_stop(self, iteration, best_history):
        if self.baseline is None:
            # Without a baseline the relative features are meaningless.  Refuse
            # rather than silently deciding on garbage.
            raise RuntimeError(
                'DQNStopper.bind_baseline() was never called; the campaign '
                'must measure the untuned configuration first')
        state = feature_module.stopper_state(
            iteration=iteration, horizon=self.horizon, baseline=self.baseline,
            best_history=best_history, sense=self.objective.sense)
        values = self.model.predict(state)
        if values[STOP] > values[CONTINUE]:
            return True, ('predicted improvement over the next {0} iterations '
                          'is below the {1:.1%}-per-iteration threshold '
                          '(Q_continue={2:.4f})'.format(
                              REWARD_DELAY, self.step_cost, values[CONTINUE]))
        return False, ''

    # -- offline training ---------------------------------------------------

    def _episodes(self, count, rng):
        curves = sample_curves(rng, count, horizon=self.horizon)
        return [Episode(curve, self.horizon, rng, self.step_cost)
                for curve in curves]

    def _targets(self, episodes, bootstrap):
        """Fitted-Q targets for every decision point.

        Continue: the discounted sum of the next REWARD_DELAY step rewards,
        plus the bootstrapped value of the state reached after them.
        Stop: zero, because stopping ends the episode.

        `bootstrap` is a frozen copy of the network -- a target network.  Without
        it the targets move while they are being fitted, the value of continuing
        feeds back into its own target, and the whole thing diverges.  That is
        not a subtle failure: the loss rises monotonically and the policy
        degenerates to always-continue.
        """
        batch = []
        for episode in episodes:
            for index in range(episode.horizon):
                total = 0.0
                discount = 1.0
                landed = index
                for offset in range(REWARD_DELAY):
                    position = index + offset
                    if position + 1 >= episode.horizon:
                        break
                    total += discount * episode.rewards[position]
                    discount *= self.discount
                    landed = position + 1
                if landed < episode.horizon - 1:
                    future = bootstrap.predict(episode.states[landed])
                    # Continuing can always be declined, so its value is never
                    # worse than stopping.  Flooring at zero keeps the target
                    # consistent with Q(stop) == 0.
                    total += discount * max(future[CONTINUE], 0.0)
                batch.append((episode.states[index], [total, 0.0], [1, 1]))
        return batch

    def _stagnated(self, scores, stops):
        """III-D's training stop condition, plus a warm-up floor.

        The paper stops "once the average reward of the agent begins to
        stagnate ... indicated by 5% or less increase across five iterations".
        Applied literally that fires immediately: for the first dozen rounds the
        policy is uniformly always-continue and its score does not move at all,
        which is flat for reasons that have nothing to do with convergence.

        So two extra conditions.  At least MIN_ROUNDS rounds must have run, and
        the policy's mean stopping time must have settled to within half an
        iteration.  The second is the operational meaning of "stagnate" here --
        the agent has stopped changing its mind.
        """
        if len(scores) <= max(STAGNATION_WINDOW, MIN_ROUNDS):
            return False
        earlier = scores[-(STAGNATION_WINDOW + 1)]
        if earlier <= 0:
            return False
        growth = (scores[-1] - earlier) / abs(earlier)
        if growth > STAGNATION_THRESHOLD:
            return False
        return abs(stops[-1] - stops[-(STAGNATION_WINDOW + 1)]) < 0.5

    def stopping_time(self, episode):
        """Where the current greedy policy would stop on this episode."""
        for index in range(episode.horizon):
            values = self.model.predict(episode.states[index])
            if values[STOP] > values[CONTINUE]:
                return index
        return episode.horizon - 1

    def _evaluate(self, episodes):
        """How close the greedy policy gets to the best stopping time.

        Scored as achieved return over best achievable return, averaged across
        the held-out curves.  1.0 means the policy stopped optimally on every
        one.  Scored in the reward's own units rather than in RoTI, because
        those are the units the agent is trained on.
        """
        ratios = []
        stops = []
        for episode in episodes:
            stop_at = self.stopping_time(episode)
            achieved = episode.return_from(0, stop_at)
            best = episode.best_return()
            ratios.append(achieved / best if best > 0 else
                          (1.0 if achieved >= 0 else 0.0))
            stops.append(stop_at)
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        mean_stop = sum(stops) / float(len(stops)) if stops else 0.0
        return mean_ratio, mean_stop

    def fit(self, curves=DEFAULT_CURVES, rounds=DEFAULT_ROUNDS,
            epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH, verbose=False):
        """Train on synthetic curves.  Returns a short report.

        Each round freezes a target network, computes targets against it, then
        takes `epochs` gradient steps on those fixed targets.  Standard fitted Q
        iteration.
        """
        rng = random.Random(self.seed)
        training = self._episodes(curves, rng)
        holdout = self._episodes(max(20, curves // 4), rng)

        bootstrap = nets.build(feature_module.STOPPER_INPUTS, 2, seed=self.seed)
        scores = []
        stops = []
        history = []
        for round_index in range(rounds):
            bootstrap.copy_from(self.model)
            batch = self._targets(training, bootstrap)
            loss = 0.0
            for _ in range(epochs):
                sample = (batch if len(batch) <= batch_size
                          else rng.sample(batch, batch_size))
                loss = self.model.fit(sample)
            score, mean_stop = self._evaluate(holdout)
            scores.append(score)
            stops.append(mean_stop)
            history.append((round_index, loss, score, mean_stop))
            if verbose:
                print('  round {0:>3}  loss {1:.6f}  return ratio {2:.3f}  '
                      'mean stop {3:.1f}/{4}'.format(
                          round_index, loss, score, mean_stop, self.horizon))
            if self._stagnated(scores, stops):
                break
        self.trained = True
        final_ratio, final_stop = self._evaluate(holdout)
        return {
            'backend': self.model.backend,
            'rounds': len(scores),
            'curves': curves,
            'return_ratio': final_ratio,
            'mean_stop': final_stop,
            'horizon': self.horizon,
            'step_cost': self.step_cost,
            'history': history,
        }
