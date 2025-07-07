import deap
from deap import tools
from deap import algorithms

#todo: check ref points in gp_specs

def nsga3(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    # Compile statistics about the population
    record = stats.compile(population)
    logbook.record(gen=0, evals=len(invalid_ind), **record)

    best_point, worst_point, extreme_points = None, None, None
    # Begin the generational process
    for gen in range(1, ngen + 1):
        offspring = deap.algorithms.varAnd(population, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population from parents and offspring
        pop, nsga3_memory = toolbox.select(population + offspring, len(population),best_point=best_point,
             worst_point=worst_point, extreme_points=extreme_points)

        best_point = nsga3_memory[0]
        worst_point = nsga3_memory[1]
        extreme_points = nsga3_memory[2]

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

    return population, logbook