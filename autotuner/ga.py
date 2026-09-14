"""The genetic algorithm both baselines search with.

H5Tuner and TunIO are both genetic searches over the same encoding; they differ
in the selection operator and in what gets bolted on around the loop.

    H5Tuner   roulette-wheel selection, elitism 3, single-point crossover 0.9,
              per-gene mutation 0.15.  Original ran on pyevolve, whose
              G1DList defaults supplied the crossover operator and rate that
              evo/evolve.py never set explicitly.
    TunIO     tournament selection -- three individuals drawn at random, the
              best two become parents -- plus elitism.  III-A says the
              combination is deliberate: elitism over-specialises the
              population and traps the search in a local optimum, and
              tournament selection is there to soften that.

Both are reproduced here rather than generalised into a pluggable optimiser.
These are baselines; their algorithms are part of what is being compared.

Subset masking
--------------
TunIO tunes a subset of the parameters each iteration.  A mask is a set of
parameter names the operators are allowed to touch; every other gene rides
along unchanged in whichever individual carries it.

The paper does not say what an excluded parameter holds -- default, current, or
best-so-far.  We freeze it at the individual's current value, which is the only
choice consistent with the paper's own statement that elitism carries the best
configuration forward: reverting to defaults would discard exactly what elitism
is preserving.  This is a reading, not a specification.
"""

ROULETTE = 'roulette'
TOURNAMENT = 'tournament'

# pyevolve's Consts.CDefGACrossoverRate, inherited by evo/evolve.py.
DEFAULT_CROSSOVER_RATE = 0.9
# ga.setMutationRate(0.15) in the original.
DEFAULT_MUTATION_RATE = 0.15
# pyevolve fed roulette through linear scaling with this multiplier.  We
# reproduce the resulting selection ratio -- best individual's share is this
# many times the worst individual's -- rather than cloning pyevolve's internal
# scaling, so scores will not match the original runs number for number.
SELECTION_PRESSURE = 1.2
# TunIO III-A: three drawn, best two become parents.
TOURNAMENT_SIZE = 3


class GeneticSearch:
    """Generational GA over index-encoded points.

    space           the parameter space
    objective       supplies `sense` and `better`
    population_size individuals per generation
    elite           individuals carried over unchanged
    selection       ROULETTE or TOURNAMENT
    mutation_rate   per-gene probability, applied to masked genes only
    crossover_rate  probability a parent pair is crossed rather than copied
    scale_mutation  when True, divide mutation_rate by the number of active
                    genes so a child mutates about one gene regardless of how
                    many parameters are in play.  Off by default: both papers
                    use a fixed per-gene rate, and turning this on is a
                    deviation from them.  It matters at high parameter counts --
                    0.15 across 21 genes mutates three genes per child, which
                    destroys building blocks faster than selection can build
                    them.  See 07-후속-연구-계획.md #5.5.
    """

    def __init__(self, space, objective, population_size, elite=0,
                 selection=ROULETTE, mutation_rate=DEFAULT_MUTATION_RATE,
                 crossover_rate=DEFAULT_CROSSOVER_RATE, scale_mutation=False,
                 rng=None):
        if selection not in (ROULETTE, TOURNAMENT):
            raise ValueError('unknown selection {0!r}'.format(selection))
        if population_size < 2:
            raise ValueError('population_size must be at least 2')
        self.space = space
        self.objective = objective
        self.population_size = population_size
        self.elite = min(elite, population_size)
        self.selection = selection
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.scale_mutation = scale_mutation
        self.rng = rng

    # -- helpers ------------------------------------------------------------

    def _active_positions(self, mask):
        if mask is None:
            return list(range(len(self.space.parameters)))
        names = set(mask)
        return [i for i, p in enumerate(self.space.parameters)
                if p.name in names]

    def _effective_mutation_rate(self, active_count):
        if not self.scale_mutation or active_count <= 0:
            return self.mutation_rate
        return 1.0 / active_count

    # -- operators ----------------------------------------------------------

    def initial_population(self, mask=None):
        return [self.space.sample(self.rng, mask=mask)
                for _ in range(self.population_size)]

    def mutate(self, point, positions, rate):
        genes = list(point)
        for position in positions:
            if self.rng.random() < rate:
                candidates = self.space.parameters[position].values
                if len(candidates) > 1:
                    genes[position] = self.rng.randrange(len(candidates))
        return tuple(genes)

    def crossover(self, mom, dad, positions):
        """Single-point crossover restricted to the active positions.

        Genes outside the mask are inherited from the parent the child is
        based on, so an inactive parameter is never shuffled by recombination.
        """
        if len(positions) < 2:
            return tuple(mom), tuple(dad)
        cut = self.rng.randrange(1, len(positions))
        sister = list(mom)
        brother = list(dad)
        for position in positions[cut:]:
            sister[position] = dad[position]
            brother[position] = mom[position]
        return tuple(sister), tuple(brother)

    def _roulette(self, scored):
        """Roulette-wheel pick from [(point, score), ...].

        Weights are linearly scaled so the best individual's share is
        SELECTION_PRESSURE times the worst individual's.  That keeps the worst
        candidate reachable instead of zeroing it out, which is what pyevolve's
        linear scaling did.
        """
        scores = [score for _, score in scored]
        if self.objective.sense == 'min':
            best, worst = min(scores), max(scores)
            raw = [worst - score for score in scores]
        else:
            best, worst = max(scores), min(scores)
            raw = [score - worst for score in scores]
        spread = abs(best - worst)
        if spread <= 0.0:
            return self.rng.choice(scored)[0]
        floor = spread / (SELECTION_PRESSURE - 1.0)
        weights = [value + floor for value in raw]
        target = self.rng.uniform(0.0, sum(weights))
        running = 0.0
        for (point, _), weight in zip(scored, weights):
            running += weight
            if running >= target:
                return point
        return scored[-1][0]

    def _tournament_pair(self, scored):
        """Draw TOURNAMENT_SIZE individuals; return the best two as parents."""
        size = min(TOURNAMENT_SIZE, len(scored))
        drawn = self.rng.sample(scored, size)
        drawn.sort(key=lambda pair: pair[1],
                   reverse=self.objective.sense == 'max')
        if len(drawn) == 1:
            return drawn[0][0], drawn[0][0]
        return drawn[0][0], drawn[1][0]

    def _parents(self, scored):
        if self.selection == TOURNAMENT:
            return self._tournament_pair(scored)
        return self._roulette(scored), self._roulette(scored)

    # -- generation step ----------------------------------------------------

    def sort_scored(self, scored):
        """Best first."""
        return sorted(scored, key=lambda pair: pair[1],
                      reverse=self.objective.sense == 'max')

    def next_population(self, scored, mask=None, elites=None):
        """Build the next generation from this one's scored individuals.

        elites, when given, is a list of (point, score) carried in unchanged --
        the caller decides what counts as an elite so a framework can keep the
        best across generations rather than within one.
        """
        positions = self._active_positions(mask)
        rate = self._effective_mutation_rate(len(positions))

        children = []
        carried = [] if not elites else [point for point, _ in
                                         self.sort_scored(elites)[:self.elite]]
        children.extend(carried)

        while len(children) < self.population_size:
            mom, dad = self._parents(scored)
            if self.rng.random() < self.crossover_rate:
                sister, brother = self.crossover(mom, dad, positions)
            else:
                sister, brother = tuple(mom), tuple(dad)
            children.append(self.mutate(sister, positions, rate))
            if len(children) < self.population_size:
                children.append(self.mutate(brother, positions, rate))

        return children[:self.population_size]
