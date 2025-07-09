import random
import pickle
import time
import copy
from deap import tools
import numpy as np
import os
import plotly.io as pio
pio.renderers.default = "browser"

import optimization_setup.deap_nsga_3
from optimization_setup.problem import test_problem_vector, comola_single_cells, comola_patches, comola_single_cells_uncertainty
from optimization_setup.gp_specs import get_deap_gp_specifications
from help_and_utility_functions.utility_functions import get_optimal_solutions
from help_and_utility_functions import plotting

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def land_use_allocation_gp(landuse_problem, algorithm, ngen, npop, inner_optimization_loops = 1,verbose_runtime_generation = True, seed = None):
    start = time.time()
    optimization_logger = {"nr_generations": ngen, "population_size": npop, "runs": {}}

    # random state for optimization
    for j in range(inner_optimization_loops):
        start_inner_loop = time.time()
        # set random state for inner loop with the same land use order
        if seed is not None:
            random.seed(j)
        pop = landuse_problem.toolbox.population(n=npop)
        hof = tools.ParetoFront()

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

        optimization_logger["runs"][j] = {"generations": {}}
        for k in range(ngen):
            gen_start = time.time()
            pop, log = algorithm(pop, landuse_problem.toolbox, 0.5, 0.1, stats=mstats,
                                           ngen=1, halloffame=hof, verbose=False)
            optimization_logger["runs"][j]["generations"][k] = {"population": copy.deepcopy(pop), "pareto_front": copy.deepcopy(hof)}
            gen_end = time.time()
            if verbose_runtime_generation:
                print("Generation {} from {} took {} minutes to compute.".format(k, ngen, (gen_end-gen_start)/60))
        end_inner_loop = time.time()
        optimization_logger["runs"][j]["run_time_minutes"] = str(int((end_inner_loop - start_inner_loop) / 60))

    end = time.time()
    print("Optimization finished after " + str(int((end - start))) + " seconds with " + str(inner_optimization_loops) + " inner loops.")
    optimization_logger["total_run_time_seconds"] = str(end - start)
    return optimization_logger


def main():
    psize = 10
    #landuse_problem = comola_single_cells((psize,psize))
    landuse_problem = comola_patches((psize,psize))

    #psize = '20_fluren'
    #landuse_problem = test_problem_vector(study_area = psize)

    # define on how many times the optimization shall be executed with different random states
    inner_optimization_loops = 1

    ngen = 100
    npop = 200

    # here, the specifications of the gp are defined.
    # toolbox: individual representation, mutation, crossover etc.
    # pset: primitive set of gp
    get_deap_gp_specifications(landuse_problem)

    algorithm = optimization_setup.deap_nsga_3.nsga3

    run_intrinsic_landuse_order = land_use_allocation_gp(landuse_problem,algorithm, ngen, npop,
                                                            inner_optimization_loops)
    with open(r'output_data\run_time_analysis\gp_{}_nsga3_{}_gens_{}_psize{}.pkl'.format(landuse_problem.encoding, ngen, npop, psize), 'wb') as handle:
        pickle.dump(run_intrinsic_landuse_order, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_time_analysis(problem_type = "raster"):
    if problem_type == "raster":
        psizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ,110 , 120, 130, 140, 150]
    elif problem_type == "vector":
        #psizes = ['10_fluren', '20_fluren','30_fluren', '40_fluren', '50_fluren','60_fluren', '70_fluren', '80_fluren', '90_fluren', '100_fluren']
        psizes = ['50_fluren']
    else:
        print("Select problem type vector or raster")

    for p_size in psizes:
        landuse_problem = None
        if problem_type == "raster":
            landuse_problem = comola_single_cells(size=(p_size, p_size))
            problem_name = str(p_size*p_size)
        elif problem_type == "vector":
            landuse_problem = test_problem_vector(p_size)
            problem_name = str(p_size)
        else:
            print("Select problem type vector or raster")

        # define on how many times the optimization shall be executed with different random states
        inner_optimization_loops = 1

        ngen = 10
        npop = 40

        # here, the specifications of the gp are defined.
        # toolbox: individual representation, mutation, crossover etc.
        # pset: primitive set of gp
        get_deap_gp_specifications(landuse_problem)

        algorithm = optimization_setup.deap_nsga_3.nsga3

        run_intrinsic_landuse_order = land_use_allocation_gp(landuse_problem, algorithm, ngen, npop,
                                                             inner_optimization_loops, verbose_runtime_generation=False)

        with open(r'output_data\run_time_analysis\gp_{}_nsga3_{}_gens_{}_popsize_problem_size{}.pkl'.format(
                landuse_problem.encoding, ngen, npop, problem_name), 'wb') as handle:
            pickle.dump(run_intrinsic_landuse_order, handle, protocol=pickle.HIGHEST_PROTOCOL)


def uncertainty_analysis():

    inner_loop_uncertainty_analysis = True
    outer_loop_uncertainty_analysis = True

    ngen = 10
    npop = 20
    p_size = 10

    if inner_loop_uncertainty_analysis:
        nr_loops = 100


        landuse_problem = comola_single_cells_uncertainty(size=(p_size, p_size), seed = np.random.RandomState(123) )

        # here, the specifications of the gp are defined.
        # toolbox: individual representation, mutation, crossover etc.
        # pset: primitive set of gp
        get_deap_gp_specifications(landuse_problem)

        algorithm = optimization_setup.deap_nsga_3.nsga3
        all_pfs = []
        for i in range(nr_loops):
            run_uncertainty = land_use_allocation_gp(landuse_problem, algorithm, ngen, npop,
                                                                 1, verbose_runtime_generation=False)
            optimal_solutions_gp = get_optimal_solutions(run_uncertainty, run=0)
            pf_gp = np.array([np.array(sol.objective_values) for sol in optimal_solutions_gp])
            all_pfs.append(pf_gp)
        plotting.plot_2d_uncertain_pareto_fronts(landuse_problem, all_pfs)
        plotting.plot_2d_uncertain_pareto_fronts2(landuse_problem, all_pfs)

    if outer_loop_uncertainty_analysis:
        nr_loops = 100
        inner_optimization_loops = 10

        # here, the specifications of the gp are defined.
        # toolbox: individual representation, mutation, crossover etc.
        # pset: primitive set of gp

        all_pfs = []
        for i in range(nr_loops):
            landuse_problem = comola_single_cells_uncertainty(size=(p_size, p_size), seed=np.random.RandomState(i))
            get_deap_gp_specifications(landuse_problem)

            algorithm = optimization_setup.deap_nsga_3.nsga3
            run_uncertainty = land_use_allocation_gp(landuse_problem, algorithm, ngen, npop,
                                                                 inner_optimization_loops, verbose_runtime_generation=False)
            optimal_solutions_gp = get_optimal_solutions(run_uncertainty, run=0)
            pf_gp = np.array([np.array(sol.objective_values) for sol in optimal_solutions_gp])
            all_pfs.append(pf_gp)
        plotting.plot_2d_uncertain_pareto_fronts(landuse_problem, all_pfs)
        plotting.plot_2d_uncertain_pareto_fronts2(landuse_problem, all_pfs)




if __name__ == "__main__":
    # with open(os.path.join('output_data', 'vector_test_10_gens.pkl'), 'rb') as handle:
    #     pf_intrinsic_land_use_order_vector = pickle.load(handle)

    #main()
    run_time_analysis(problem_type="vector")
    uncertainty_analysis()