"""The generational loop the baselines share.

Both H5Tuner and TunIO are generational genetic searches.  Factoring the loop
out is not an attempt to make them interchangeable -- their operators, stoppers
and subset policies stay fixed by their own modules -- but it does guarantee
they see identical evaluation, identical bookkeeping and identical stopping
mechanics.  If the loop differed, a difference in results would not be
attributable to the algorithms.
"""

from .. import ga as ga_module


class Framework:
    """Base class.  A framework configures a search and runs it."""

    name = None
    description = ''

    def run(self):
        raise NotImplementedError


class GeneticFramework(Framework):
    """A generational GA, plus an optional stopper and subset selector.

    space, objective, evaluator, campaign  the shared substrate
    generations       maximum iterations
    population        individuals per generation
    elite             individuals carried across generations unchanged
    selection         ga.ROULETTE or ga.TOURNAMENT
    mutation_rate     per-gene probability
    crossover_rate    probability a parent pair is recombined
    scale_mutation    divide mutation_rate by the active gene count
    stopper           early_stopping.Stopper
    subset_selector   subset.SubsetSelector
    reporter          optional callable(event, payload) for progress output
    """

    def __init__(self, space, objective, evaluator, campaign, rng,
                 generations, population, elite, selection, mutation_rate,
                 crossover_rate, scale_mutation, stopper, subset_selector,
                 reporter=None):
        self.space = space
        self.objective = objective
        self.evaluator = evaluator
        self.campaign = campaign
        self.rng = rng
        self.generations = generations
        self.stopper = stopper
        self.subset_selector = subset_selector
        self.reporter = reporter
        self.search = ga_module.GeneticSearch(
            space=space, objective=objective, population_size=population,
            elite=elite, selection=selection, mutation_rate=mutation_rate,
            crossover_rate=crossover_rate, scale_mutation=scale_mutation,
            rng=rng)

    # -- reporting ----------------------------------------------------------

    def _emit(self, event, **payload):
        if self.reporter is not None:
            self.reporter(event, payload)

    # -- loop ---------------------------------------------------------------

    def run(self):
        campaign = self.campaign
        objective = self.objective

        # RoTI needs the untuned configuration's score, and measuring it costs
        # budget like anything else.
        baseline = self.evaluator.baseline()
        campaign.set_baseline(baseline)
        campaign.record(baseline, self.evaluator.tuning_seconds, iteration=-1)
        # Learned components normalise everything against the untuned score, so
        # they cannot act before this point.
        if hasattr(self.stopper, 'bind_baseline'):
            self.stopper.bind_baseline(baseline.score)
        self._emit('baseline', result=baseline)

        mask = self.subset_selector.select(0, campaign)
        population = self.search.initial_population(mask=mask)
        hall = []
        best_history = []

        for iteration in range(self.generations):
            if iteration > 0:
                mask = self.subset_selector.select(iteration, campaign)

            self._emit('generation_start', iteration=iteration, mask=mask)

            # One call for the whole generation.  The candidates are still
            # evaluated one after another -- the loop below sees them in the
            # same order with the same scores -- but a launcher that can batch
            # submits them together, which under a busy scheduler is the
            # difference between one queue wait per generation and one per
            # candidate.
            scored = []
            results = self.evaluator.evaluate_many(
                population, tag='gen-{0:03d}'.format(iteration))
            for point, result in zip(population, results):
                campaign.record(result, self.evaluator.tuning_seconds,
                                iteration)
                scored.append((point, result.score))
                self._emit('evaluated', iteration=iteration, result=result)

            ordered = self.search.sort_scored(scored)
            hall = self.search.sort_scored(hall + ordered)[:max(
                1, self.search.elite)]

            campaign.close_generation(
                index=iteration,
                tuning_seconds=self.evaluator.tuning_seconds,
                scores=[score for _, score in scored],
                evaluations=len(scored),
                subset=mask)
            best_history.append(campaign.best_score)

            self.subset_selector.observe(iteration, mask, campaign)
            self._emit('generation_end', iteration=iteration,
                       generation=campaign.generations[-1])

            stop, reason = self.stopper.should_stop(iteration, best_history)
            if stop:
                campaign.stop(iteration, reason)
                self._emit('stopped', iteration=iteration, reason=reason)
                break

            if iteration < self.generations - 1:
                population = self.search.next_population(
                    ordered, mask=mask, elites=hall)

        if hasattr(self.subset_selector, 'finish'):
            self.subset_selector.finish(campaign)
        return campaign
