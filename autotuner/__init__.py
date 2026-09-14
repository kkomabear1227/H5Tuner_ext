"""Multi-layer parallel I/O autotuning: baseline frameworks for comparison.

Two published frameworks are reimplemented against a single shared substrate so
their results can be compared on equal footing:

    h5tuner           genetic search, no early stopping        (Behzad et al.)
    h5tuner-original  the historical evo/evolve.py behaviour, bug-for-bug
    tunio             genetic search + RL subset selection
                      + RL early stopping + I/O kernel         (IPDPS'24)

Each framework keeps the search methodology its paper specifies.  H5Tuner is a
genetic algorithm; TunIO is a genetic algorithm with two reinforcement-learning
components bolted on.  They are not interchangeable knobs here -- swapping
methodologies is a question for the new framework, not for these baselines.

What the frameworks share, so that a comparison means something:

    space           the parameter space every framework searches
    config_writer   render a candidate as H5Tuner's config.xml
    evaluate        launch the application and measure it
    objective       what "better" means (wall-clock vs bandwidth)
    trace           record/replay measurements so search can be tuned offline
    metrics         RoTI and friends

Only `h5tuner-original` deviates: it reproduces the historical objective and
the hand-collapsed parameter space, and is provided for reference rather than
for the head-to-head comparison.
"""

__all__ = [
    'config_writer',
    'evaluate',
    'metrics',
    'objective',
    'space',
    'trace',
]
