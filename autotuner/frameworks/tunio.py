"""TunIO: genetic search plus RL subset selection and RL early stopping.

Rajesh, Bateman, Bez, Byna, Kougkas, Sun.  "TunIO: An AI-powered Framework for
Optimizing HPC I/O."  IPDPS 2024, pp.494-504.  DOI 10.1109/IPDPS57955.2024.00050

TunIO does not replace H5Tuner; it wraps it.  The C shim in src/ is TunIO's
parameter injection layer unchanged (III-A).  What TunIO replaces is the search
driver, and it adds three components:

    Application I/O Discovery   preprocessing: reduce the application to an I/O
                                kernel so each evaluation is cheaper.  NOT
                                IMPLEMENTED.  It sits outside the search loop --
                                it changes what the evaluator launches, not how
                                candidates are chosen -- so the framework runs
                                without it, against the full application.
    Smart Configuration         per iteration: pick a subset of parameters to
    Generation                  tune, ordered by impact.  subset.py
    Early Stopping              per iteration: decide whether to continue.
                                early_stopping.py

Search settings from III-A:

    DEAP, elitism, and tournament selection -- three individuals drawn at
    random, the best two become parents.  The paper is explicit about why the
    two are combined: elitism over-specialises the population and traps the
    search in a local optimum, and tournament selection is the mitigation.

The paper does not state population size, generation count, mutation rate or
crossover rate.  We inherit H5Tuner's, which is also what makes the comparison
clean: the same budget shape, different components.

Generations are fixed at 40, matching H5Tuner, so both frameworks get the same
budget.  IV-C runs HACC for 50 generations, but using 50 here would hand TunIO a
quarter more evaluations than the baseline and confound the comparison.

Implementation status
---------------------
Both RL components are implemented -- rl/picker.py and rl/stopper.py -- and the
design decisions the paper left open are recorded in 06-진행-상황.md 3.

One caveat carries into any result.  The early stopper is trained offline on
synthetic tuning curves whose per-iteration cost is uniform.  On real hardware it
is not: a badly configured candidate runs slower, so early iterations cost more.
Under uniform cost the paper's own heuristic baseline turns out to be near
optimal for the reward, and the learned stopper does not beat it on synthetic
curves.  Whether it does on real runs is untested.  06-진행-상황.md 4.1
has the analysis and the proposed fix.

Application I/O Discovery is not implemented, so `tunio` evaluates the full
application like the baselines do.  That makes the comparison of *search*
behaviour clean and leaves TunIO's evaluation-cost reduction out of scope.

Useful ablation arms, all runnable:

    --stopper heuristic --subset rl    isolates the learned stopper
    --stopper rl --subset static       isolates the learned ranking
    --stopper never --subset all       H5Tuner with tournament selection,
                                       isolating the selection operator alone
"""

from .. import ga as ga_module
from .base import GeneticFramework

DEFAULTS = {
    'population': 15,
    'generations': 40,
    'elite': 3,
    'mutation_rate': 0.15,
    'crossover_rate': 0.9,
    'reps': 3,
    'reduction': 'mean',
    'objective': 'bandwidth',
    'space': 'paper12',
    'stopper': 'rl',
    'subset': 'rl',
}


class TunIO(GeneticFramework):

    name = 'tunio'
    description = ('genetic search with tournament selection, RL subset '
                   'selection and RL early stopping')

    def __init__(self, space, objective, evaluator, campaign, rng,
                 generations=DEFAULTS['generations'],
                 population=DEFAULTS['population'],
                 elite=DEFAULTS['elite'],
                 mutation_rate=DEFAULTS['mutation_rate'],
                 crossover_rate=DEFAULTS['crossover_rate'],
                 scale_mutation=False, stopper=None, subset_selector=None,
                 reporter=None):
        if stopper is None or subset_selector is None:
            raise ValueError(
                'TunIO requires an explicit stopper and subset selector')
        super().__init__(
            space=space, objective=objective, evaluator=evaluator,
            campaign=campaign, rng=rng, generations=generations,
            population=population, elite=elite,
            selection=ga_module.TOURNAMENT,
            mutation_rate=mutation_rate, crossover_rate=crossover_rate,
            scale_mutation=scale_mutation, stopper=stopper,
            subset_selector=subset_selector, reporter=reporter)
