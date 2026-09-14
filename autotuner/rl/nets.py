"""Function approximators, with and without PyTorch.

One interface, two implementations.  Both map a fixed-length feature vector to
`outputs` numbers and learn from (features, target, mask) batches, where the
mask says which outputs a given sample supplies a target for.  Masking matters
for both agents: a DQN only has a target for the action it took, and the subset
picker only learns about the parameters it actually tuned.

    MLPModel        PyTorch.  Two hidden layers of `hidden` units, ReLU, Adam.
    RandomFeatures  no dependencies.  A frozen random ReLU layer feeding a
                    trained linear head.

The fallback is a random-feature approximation, not a linear model: the hidden
layer is nonlinear, it is simply not trained.  For problems this small, with
features already engineered to carry the relevant structure, the difference is
mostly sample efficiency rather than reachable policy.  It is still a weaker
approximator, so anything published from a fallback run should say so.
"""

import math
import random


# Adam on an MLP and plain SGD on a linear head want very different step sizes.
# Measured on the stopper's fitted-Q task: at 1e-3 the fallback needs several
# hundred rounds to move off always-continue; at 5e-2 it converges in about 25
# and reaches a 0.955 return ratio against the optimal stopping time.
TORCH_LEARNING_RATE = 1e-3
FALLBACK_LEARNING_RATE = 5e-2


def build(inputs, outputs, hidden=64, seed=None, learning_rate=None):
    """Return the best available model for this environment."""
    from . import HAS_TORCH
    if HAS_TORCH:
        return MLPModel(inputs, outputs, hidden=hidden, seed=seed,
                        learning_rate=learning_rate or TORCH_LEARNING_RATE)
    return RandomFeatures(inputs, outputs, hidden=hidden, seed=seed,
                          learning_rate=learning_rate
                          or FALLBACK_LEARNING_RATE)


class Model:
    """Interface both implementations honour."""

    backend = None

    def predict(self, features):
        """features: list[float] -> list[float] of length `outputs`."""
        raise NotImplementedError

    def fit(self, batch):
        """batch: iterable of (features, targets, mask).  Returns mean loss.

        `targets` and `mask` are both length `outputs`; entries where mask is
        falsy contribute nothing to the loss.
        """
        raise NotImplementedError

    def copy_from(self, other):
        """Adopt another model's weights.  Used for DQN target networks."""
        raise NotImplementedError


class RandomFeatures(Model):
    """Frozen random ReLU layer, trained linear head.

    phi(x) = [relu(W x + b), x, 1]

    The head is trained by plain SGD on masked squared error.  Everything is
    deterministic given `seed`, which matters because reproducing a tuning run
    is otherwise impossible.
    """

    backend = 'fallback'

    def __init__(self, inputs, outputs, hidden=64, seed=None,
                 learning_rate=FALLBACK_LEARNING_RATE):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = hidden
        self.learning_rate = learning_rate

        rng = random.Random(seed if seed is not None else 0)
        # He-style scaling keeps the pre-activation variance stable, so the
        # ReLU layer neither saturates at zero nor blows up.
        scale = math.sqrt(2.0 / max(1, inputs))
        self._w = [[rng.gauss(0.0, scale) for _ in range(inputs)]
                   for _ in range(hidden)]
        self._b = [rng.gauss(0.0, 0.1) for _ in range(hidden)]
        self._width = hidden + inputs + 1
        self._head = [[0.0] * self._width for _ in range(outputs)]

    def _phi(self, features):
        expanded = []
        for row, bias in zip(self._w, self._b):
            total = bias
            for weight, value in zip(row, features):
                total += weight * value
            expanded.append(total if total > 0.0 else 0.0)
        expanded.extend(features)
        expanded.append(1.0)
        return expanded

    def predict(self, features):
        phi = self._phi(features)
        return [sum(w * p for w, p in zip(row, phi)) for row in self._head]

    def fit(self, batch):
        batch = list(batch)
        if not batch:
            return 0.0
        total_loss = 0.0
        counted = 0
        rate = self.learning_rate / len(batch)
        for features, targets, mask in batch:
            phi = self._phi(features)
            for index in range(self.outputs):
                if not mask[index]:
                    continue
                prediction = sum(w * p
                                 for w, p in zip(self._head[index], phi))
                error = prediction - targets[index]
                total_loss += error * error
                counted += 1
                step = 2.0 * rate * error
                row = self._head[index]
                for position, value in enumerate(phi):
                    row[position] -= step * value
        return total_loss / counted if counted else 0.0

    def copy_from(self, other):
        self._head = [row[:] for row in other._head]      # noqa: SLF001
        self._w = [row[:] for row in other._w]            # noqa: SLF001
        self._b = other._b[:]                             # noqa: SLF001


class MLPModel(Model):
    """Two hidden layers, ReLU, Adam, masked mean squared error.

    This is the "shared trunk plus head" reading of the paper's State Observer
    and Action Decider: the hidden layers are the learned state representation
    and the output layer is the decision, trained end to end.
    """

    backend = 'torch'

    def __init__(self, inputs, outputs, hidden=64, seed=None,
                 learning_rate=TORCH_LEARNING_RATE):
        import torch
        from torch import nn

        if seed is not None:
            torch.manual_seed(seed)
        self._torch = torch
        self.inputs = inputs
        self.outputs = outputs
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, outputs))
        self.optimiser = torch.optim.Adam(self.net.parameters(),
                                          lr=learning_rate)

    def predict(self, features):
        torch = self._torch
        with torch.no_grad():
            tensor = torch.tensor([features], dtype=torch.float32)
            return self.net(tensor)[0].tolist()

    def fit(self, batch):
        torch = self._torch
        batch = list(batch)
        if not batch:
            return 0.0
        features = torch.tensor([row[0] for row in batch], dtype=torch.float32)
        targets = torch.tensor([row[1] for row in batch], dtype=torch.float32)
        mask = torch.tensor([[1.0 if flag else 0.0 for flag in row[2]]
                             for row in batch], dtype=torch.float32)
        predictions = self.net(features)
        squared = ((predictions - targets) ** 2) * mask
        denominator = mask.sum().clamp(min=1.0)
        loss = squared.sum() / denominator
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return float(loss.item())

    def copy_from(self, other):
        self.net.load_state_dict(other.net.state_dict())
