"""H5Tuner: genetic search over the I/O stack, no early stopping.

Behzad et al., "A framework for auto-tuning HDF5 applications" (HPDC'13) and
"Auto-Tuning of Parallel I/O Parameters for HDF5 Applications".  The tuner that
ships in this repository as evo/evolve.py.

This is the *fair-comparison* arm.  It runs H5Tuner's search algorithm over the
same parameter space, the same objective and the same measurement pipeline as
TunIO, which is how the TunIO paper's own IV-D comparison is set up: the
"H5Tuner with No Stop" and "H5Tuner with Heuristic Stop" rows differ from TunIO
only in the three components, not in what is being measured.

For the historical behaviour -- wall-clock objective, the hand-collapsed
five-gene space, 40 fixed generations -- use h5tuner-original instead.

Settings come from evo/evolve.py, which took them from pyevolve:

    population 15         ga.setPopulationSize(15)
    generations 40        ga.setGenerations(40)
    elite 3               ga.setElitismReplacement(3)
    mutation 0.15         ga.setMutationRate(0.15)
    crossover 0.9         pyevolve's G1DList default, never set explicitly
    roulette wheel        GRouletteWheel

No early stopping.  evo/evolve.py defined a ConvergenceCriteria but the line
registering it was commented out, so every run went the full 40 generations.
"""

from .. import ga as ga_module
from .. import subset as subset_module
from ..early_stopping import NeverStop
from .base import GeneticFramework

DEFAULTS = {
    'population': 15,
    'generations': 40,
    'elite': 3,
    'mutation_rate': 0.15,
    'crossover_rate': 0.9,
    'reps': 5,
    'objective': 'bandwidth',
    'space': 'paper12',
    'stopper': 'never',
    'subset': 'all',
}


class H5Tuner(GeneticFramework):

    name = 'h5tuner'
    description = 'genetic search, roulette-wheel selection, no early stopping'

    def __init__(self, space, objective, evaluator, campaign, rng,
                 generations=DEFAULTS['generations'],
                 population=DEFAULTS['population'],
                 elite=DEFAULTS['elite'],
                 mutation_rate=DEFAULTS['mutation_rate'],
                 crossover_rate=DEFAULTS['crossover_rate'],
                 scale_mutation=False, stopper=None, subset_selector=None,
                 reporter=None):
        super().__init__(
            space=space, objective=objective, evaluator=evaluator,
            campaign=campaign, rng=rng, generations=generations,
            population=population, elite=elite,
            selection=ga_module.ROULETTE,
            mutation_rate=mutation_rate, crossover_rate=crossover_rate,
            scale_mutation=scale_mutation,
            stopper=stopper if stopper is not None else NeverStop(),
            subset_selector=(subset_selector if subset_selector is not None
                             else subset_module.AllParameters()),
            reporter=reporter)
