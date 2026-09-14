"""H5Tuner as evo/evolve.py actually shipped it.

Provided for reference, not for the head-to-head comparison.  It searches a
different space and optimises a different quantity than TunIO, so its scores are
not comparable to anything else here.  What it is good for is showing how much
of the reported gap comes from the algorithm and how much from the experiment
setup being different.

What makes it "original":

    space = original      five declared genes, but only four degrees of
                          freedom.  cb_buffer_size is tied to striping_unit
                          ("Ruth(David's) Suggestion") and sieve_buf_size is
                          pinned to a single 512MB value.  alignment keeps its
                          original paired values, capped at 256KB, including the
                          redundant threshold 0/1 pair.  13,440 configurations
                          against paper12's hundreds of millions.

    objective = walltime  the application's wall-clock time, minimised.  Not an
                          I/O measurement: it includes compute.

    reduction = min       fitness is the fastest of the repetitions.  The
                          original ran five repetitions and then used only the
                          last one, because the loop overwrote `elapsed` instead
                          of reducing it; min() is the fix applied when
                          evolve.py was rewritten for Python 3.

    no early stopping     40 generations, always.

Deliberately NOT reproduced, because they are defects rather than design:

    the config.xml path bug   home_dir + 'config.xml' with no separator, so the
                              file landed next to the home directory and the
                              shim never saw it.  Reproducing this would make
                              every candidate score identical noise.
    the cleanup that never ran  Popen(["cmd", "rm SDS.h5"], shell=True) runs only
                              the first element, so output files survived
                              between candidates.
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
    'reduction': 'min',
    'objective': 'walltime',
    'space': 'original',
    'stopper': 'never',
    'subset': 'all',
}


class H5TunerOriginal(GeneticFramework):

    name = 'h5tuner-original'
    description = ('evo/evolve.py as shipped: wall-clock objective, '
                   'four effective dimensions, 40 fixed generations')

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
