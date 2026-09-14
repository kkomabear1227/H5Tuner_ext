"""Reinforcement-learning components for TunIO.

Two agents, both reconstructed from III-C and III-D:

    stopper   decides whether to continue tuning.  A DQN over an engineered
              state, trained offline on synthetic logarithmic tuning curves.
    picker    decides which parameters to tune this iteration.  A contextual
              bandit that scores each parameter and takes the top k.

Both run on PyTorch when it is installed and on a dependency-free fallback
otherwise.  The fallback is not a stub: it trains on the same data, with the
same reward and the same loop, using a frozen random hidden layer and a trained
linear head.  That is a weaker function approximator than a full MLP, and
`BACKEND` says which one produced a given result so it can be reported.

Design decisions taken here are recorded in 06-진행-상황.md.  Two are
load-bearing and were chosen deliberately over alternatives:

  * the picker scores parameters individually and takes the top k, rather than
    choosing among all 2^n subsets.  Learning 4,096 action values from 40
    iterations is not possible; learning n parameter scores is.
  * the stopper folds the paper's State Observer and Action Decider into one
    network trained end to end, which is a standard DQN.  At this problem size
    (state of 8 dimensions, two actions) the architecture does not change the
    outcome, so the simpler and more describable choice wins.
"""

try:                                            # pragma: no cover - env probe
    import torch                                # noqa: F401
    BACKEND = 'torch'
except ImportError:                             # pragma: no cover - env probe
    BACKEND = 'fallback'

HAS_TORCH = BACKEND == 'torch'

__all__ = ['BACKEND', 'HAS_TORCH', 'features', 'nets', 'picker', 'stopper']
