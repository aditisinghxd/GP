import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from help_and_utility_functions import plotting
from help_and_utility_functions.CoMOLA_help_functions import get_optimal_solutions_GA_CoMOLA
from optimization_setup.gp_specs import get_deap_gp_specifications
from help_and_utility_functions.utility_functions import get_optimal_solutions, get_optimal_solutions_pymoo
from optimization_setup.problem import test_problem_vector, comola_single_cells, comola_patches
from optimization_setup.map_translation_and_validation import validate_landuse_map
#parDir = os.path.abspath(os.pardir)
parDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def compute_objective_values_initial_solution(landuse_problem):
    validity, landuse_map, landuse_array = validate_landuse_map(landuse_problem.initial_landuse_map_datasource, landuse_problem.constraints, landuse_column_name = "landuse_re")
    # evaluate solution for all objectives
    objective_values = [obj.formula(landuse_map=landuse_map, landuse_array=landuse_array) for obj in
                            landuse_problem.objectives]
    return objective_values

def vector_problem_run_times():
    landuse_problem = test_problem_vector('10_fluren')
    get_deap_gp_specifications(landuse_problem)
    with open(os.path.join(parDir,'output_data','run_time_analysis', 'gp_cell_nsga3_100_gens_200_popsize_problem_size100.pkl'), 'rb') as handle:
       time_100_cells = pickle.load(handle)

    get_deap_gp_specifications(landuse_problem)
    with open(os.path.join(parDir,'output_data','run_time_analysis', 'gp_cell_nsga3_100_gens_200_popsize_problem_size10000.pkl'), 'rb') as handle:
       time_10000_cells = pickle.load(handle)

    with open(os.path.join(parDir,'output_data','run_time_analysis', 'gp_cell_nsga3_100_gens_200_popsize_problem_size14400.pkl'), 'rb') as handle:
       time_14400_ccells = pickle.load(handle)

def gp_vector_small():
    landuse_problem = test_problem_vector('10_fluren')
    get_deap_gp_specifications(landuse_problem)
    with open(os.path.join(parDir, 'output_data', 'run_time_analysis',
                           'gp_vector_nsga3_100_gens_200_psize10_fluren.pkl'), 'rb') as handle:
        gp_vector_large = pickle.load(handle)
    optimal_solutions_gp = get_optimal_solutions(gp_vector_large, run=0)

    pf_gp_vector = np.array([np.array(sol.objective_values) for sol in optimal_solutions_gp])
    max_single_objective_gp = [np.max(pf_gp_vector[:, 0]), np.max(pf_gp_vector[:, 1]),
                               np.max(pf_gp_vector[:, 2]), np.max(pf_gp_vector[:, 3])]
    known_optima = np.array([o.extremes["extreme_best"] for o in landuse_problem.objectives])
    print("single objective best obtained with GP: " + str(max_single_objective_gp))
    print("Known single objective optima: " + str(known_optima))
    plotting.plot_single_objective_extremes(landuse_problem, optimal_solutions=optimal_solutions_gp,
                                            optimal_solutions_objective_values=pf_gp_vector, objectives=[0, 1, 2, 3])
    plotting.plot_2d_pareto_fronts(landuse_problem, pf_gp_vector)


def gp_vector_problem_large():
    landuse_problem = test_problem_vector('100_fluren')
    get_deap_gp_specifications(landuse_problem)
    with open(os.path.join(parDir,'output_data','run_time_analysis', 'gp_vector_nsga3_100_gens_200_psize100_fluren.pkl'), 'rb') as handle:
       gp_vector_large = pickle.load(handle)
    optimal_solutions_gp = get_optimal_solutions(gp_vector_large, run=0)
    
    pf_gp_vector = np.array([np.array(sol.objective_values) for sol in optimal_solutions_gp])
    max_single_objective_gp = [np.max(pf_gp_vector[:, 0]), np.max(pf_gp_vector[:, 1]),
                               np.max(pf_gp_vector[:, 2]), np.max(pf_gp_vector[:, 3])]
    known_optima = np.array( [o.extremes["extreme_best"] for o in landuse_problem.objectives])
    print("single objective best obtained with GP: " + str(max_single_objective_gp))
    print("Known single objective optima: " + str(known_optima))
    plotting.plot_single_objective_extremes(landuse_problem, optimal_solutions = optimal_solutions_gp,
                                                  optimal_solutions_objective_values= pf_gp_vector, objectives = [0,1,2,3])
    plotting.plot_2d_pareto_fronts(landuse_problem, pf_gp_vector)

def benchmark_nsga3_gp_vector_problem_large():
    landuse_problem = test_problem_vector('100_fluren')
    landuse_problem.objectives[3].extremes["extreme_best"] = landuse_problem.objectives[3].extremes["extreme_best"]/10
    get_deap_gp_specifications(landuse_problem)
    with open(os.path.join(parDir,'output_data','run_time_analysis', 'gp_vector_nsga3_100_gens_200_psize100_fluren.pkl'), 'rb') as handle:
       gp_vector = pickle.load(handle)

    with open(os.path.join(parDir,'output_data','run_time_analysis', 'ga_vector_nsga3_100_gens_200_p_size100_fluren.pkl'), 'rb') as handle:
       nsga3_vector = pickle.load(handle)
    optimal_solutions_gp = get_optimal_solutions(gp_vector, run=0)
    pf_gp_vector = np.array([np.array(sol.objective_values) for sol in optimal_solutions_gp])
    #to make it hectares
    pf_gp_vector[:,3] = pf_gp_vector[:,3] / 10
    plotting.plot_single_objective_extremes(landuse_problem, optimal_solutions = optimal_solutions_gp,
                                                optimal_solutions_objective_values= pf_gp_vector, objectives = [0,1,2,3])

    optimal_solutions_ga  = get_optimal_solutions_pymoo(nsga3_vector, run=0)
    pf_ga_vector = np.array([np.array(sol.objective_values) for sol in optimal_solutions_ga])
    # to make it hectares
    pf_ga_vector[:, 3] = pf_ga_vector[:, 3] / 10
    plotting.plot_2d_pareto_fronts_benchmark(landuse_problem, pf_gp_vector, pf_ga_vector)

    max_single_objective_gp = [np.max(pf_gp_vector[:, 0]), np.max(pf_gp_vector[:, 1]),
                               np.max(pf_gp_vector[:, 2]), np.max(pf_gp_vector[:, 3])]
    max_single_objective_ga = [np.max(pf_ga_vector[:, 0]), np.max(pf_ga_vector[:, 1]),
                               np.max(pf_ga_vector[:, 2]), np.max(pf_ga_vector[:, 3])]

    known_optima = np.array([o.extremes["extreme_best"] for o in landuse_problem.objectives])

    print("single objective best obtained with GP: " + str(max_single_objective_gp))
    print("single objective best obtained with NSGA 3: " + str(max_single_objective_ga))
    print("Known single objective optima: " + str(known_optima))
    plotting.plot_single_objective_extremes(landuse_problem, optimal_solutions=optimal_solutions_gp,
                                            optimal_solutions_objective_values=pf_gp_vector, objectives=[0, 1, 2, 3])

def benchmark_nsga3_gp_vector_problem_small():
    landuse_problem = test_problem_vector('10_fluren')
    landuse_problem.objectives[3].extremes["extreme_best"] = landuse_problem.objectives[3].extremes["extreme_best"]/10
    get_deap_gp_specifications(landuse_problem)
    with open(os.path.join(parDir,'output_data','run_time_analysis', 'gp_vector_nsga3_100_gens_200_psize10_fluren.pkl'), 'rb') as handle:
       gp_vector = pickle.load(handle)

    with open(os.path.join(parDir,'output_data','run_time_analysis', 'ga_vector_nsga3_100_gens_200_p_size10_fluren.pkl'), 'rb') as handle:
       nsga3_vector = pickle.load(handle)
    optimal_solutions_gp = get_optimal_solutions(gp_vector, run=0)
    pf_gp_vector = np.array([np.array(sol.objective_values) for sol in optimal_solutions_gp])
    #to make it hectares
    pf_gp_vector[:,3] = pf_gp_vector[:,3] / 10
    plotting.plot_single_objective_extremes(landuse_problem, optimal_solutions = optimal_solutions_gp,
                                                optimal_solutions_objective_values= pf_gp_vector, objectives = [0,1,2,3])

    optimal_solutions_ga  = get_optimal_solutions_pymoo(nsga3_vector, run=0)
    pf_ga_vector = np.array([np.array(sol.objective_values) for sol in optimal_solutions_ga])
    # to make it hectares
    pf_ga_vector[:, 3] = pf_ga_vector[:, 3] / 10
    plotting.plot_2d_pareto_fronts_benchmark(landuse_problem, pf_gp_vector, pf_ga_vector)

    max_single_objective_gp = [np.max(pf_gp_vector[:, 0]), np.max(pf_gp_vector[:, 1]),
                               np.max(pf_gp_vector[:, 2]), np.max(pf_gp_vector[:, 3])]
    max_single_objective_ga = [np.max(pf_ga_vector[:, 0]), np.max(pf_ga_vector[:, 1]),
                               np.max(pf_ga_vector[:, 2]), np.max(pf_ga_vector[:, 3])]

    known_optima = np.array([o.extremes["extreme_best"] for o in landuse_problem.objectives])

    print("single objective best obtained with GP: " + str(max_single_objective_gp))
    print("single objective best obtained with NSGA 3: " + str(max_single_objective_ga))
    print("Known single objective optima: " + str(known_optima))
    plotting.plot_single_objective_extremes(landuse_problem, optimal_solutions=optimal_solutions_gp,
                                            optimal_solutions_objective_values=pf_gp_vector, objectives=[0, 1, 2, 3])


def plot_run_time_progression():
    plotting.plot_run_times(os.path.join(parDir,"output_data","run_time_analysis", "run_times.csv"))

def run_time_comparison_GP_Comola():
    plotting.plot_run_time_comparison_GP_CoMOLA(os.path.join(parDir,"output_data","run_time_analysis", "run_times_with_comola.csv"))

def run_time_comparison_GP_NSGA3():
    plotting.plot_run_time_comparison_GP_NSGA3(os.path.join(parDir,"output_data","run_time_analysis", "run_times_vector.csv"))

def gp_raster_problem_large():
    landuse_problem = comola_single_cells(size = (100,100))
    get_deap_gp_specifications(landuse_problem)
    with open(os.path.join(parDir, 'output_data', 'run_time_analysis',
                           'gp_cell_nsga3_100_gens_200_psize100.pkl'), 'rb') as handle:
        gp_raster_large = pickle.load(handle)
    optimal_solutions_gp = get_optimal_solutions(gp_raster_large, run=0)

    pf_gp_raster_large = np.array([np.array(sol.objective_values) for sol in optimal_solutions_gp])
    max_single_objective_gp = [np.max(pf_gp_raster_large[:, 0]), np.max(pf_gp_raster_large[:, 1]),
                               np.max(pf_gp_raster_large[:, 2]), np.max(pf_gp_raster_large[:, 3])]
    known_optima = np.array([o.extremes["extreme_best"] for o in landuse_problem.objectives])
    print("single objective best obtained with GP: " + str(max_single_objective_gp))
    print("Known single objective optima: " + str(known_optima))
    plotting.plot_single_objective_extremes(landuse_problem, optimal_solutions=optimal_solutions_gp,
                                            optimal_solutions_objective_values=pf_gp_raster_large, objectives=[0, 1, 2, 3])
    plotting.plot_2d_pareto_fronts(landuse_problem, pf_gp_raster_large)

def gp_raster_problem_largest():
    landuse_problem = comola_single_cells(size = (1000,1000))
    get_deap_gp_specifications(landuse_problem)
    with open(os.path.join(parDir, 'output_data', 'run_time_analysis',
                           'gp_cell_nsga3_100_gens_200_psize1000.pkl'), 'rb') as handle:
        gp_raster_large = pickle.load(handle)
    optimal_solutions_gp = get_optimal_solutions(gp_raster_large, run=0)

    pf_gp_raster_large = np.array([np.array(sol.objective_values) for sol in optimal_solutions_gp])
    max_single_objective_gp = [np.max(pf_gp_raster_large[:, 0]), np.max(pf_gp_raster_large[:, 1]),
                               np.max(pf_gp_raster_large[:, 2]), np.max(pf_gp_raster_large[:, 3])]
    known_optima = np.array([o.extremes["extreme_best"] for o in landuse_problem.objectives])
    print("single objective best obtained with GP: " + str(max_single_objective_gp))
    print("Known single objective optima: " + str(known_optima))
    plotting.plot_single_objective_extremes(landuse_problem, optimal_solutions=optimal_solutions_gp,
                                            optimal_solutions_objective_values=pf_gp_raster_large, objectives=[0, 1, 2, 3])
    plotting.plot_2d_pareto_fronts(landuse_problem, pf_gp_raster_large)

def benchmark_comola():
    landuse_problem = comola_single_cells(size = (10,10))
    get_deap_gp_specifications(landuse_problem)
    with open(os.path.join(parDir,'output_data','run_time_analysis', 'gp_cell_nsga3_100_gens_200_popsize_problem_size100.pkl'), 'rb') as handle:
       gp_10_10 = pickle.load(handle)
    optimal_solutions_gp_10_10 = get_optimal_solutions(gp_10_10, run=0)
    pf_intrinsic_land_use_order = np.nan_to_num(np.array([np.array(sol.objective_values) for sol in optimal_solutions_gp_10_10]))
    max_single_objective_gp = [np.max(pf_intrinsic_land_use_order[:,0]),np.max(pf_intrinsic_land_use_order[:,1]),np.max(pf_intrinsic_land_use_order[:,2]),np.max(pf_intrinsic_land_use_order[:,3])]
    with open(os.path.join(parDir,'output_data','comola','comola_best_solutions.csv'), 'rb') as handle:
       comola_raw = np.loadtxt(handle, delimiter=',', dtype=list, skiprows=1)

    optimal_solutions_CoMOLA_GA = get_optimal_solutions_GA_CoMOLA(comola_raw)
    pf_CoMOLA_GA = np.array([np.array(sol.objective_values) for sol in optimal_solutions_CoMOLA_GA])

    max_single_objective_comola = [np.max(pf_CoMOLA_GA[:, 0]), np.max(pf_CoMOLA_GA[:, 1]),
                            np.max(pf_CoMOLA_GA[:, 2]), np.max(pf_CoMOLA_GA[:, 3])]

    known_optima = np.array([o.extremes["extreme_best"] for o in landuse_problem.objectives])
    print("single objective best obtained with GP: " + str(max_single_objective_gp))
    print("single objective best obtained with CoMOLA: " + str(max_single_objective_comola))
    print("Known single objective optima: " + str(known_optima))
    
    plotting.plot_2d_pareto_fronts_benchmark(landuse_problem, pf_intrinsic_land_use_order, pf_CoMOLA_GA)

#not displayed in paper - largest described problem with 1000 * 1000 cells of raster problem
gp_raster_problem_largest()

#Figure 5a), Table 3
benchmark_comola()

#Figure 5b), Table 3
gp_raster_problem_large()

#Figure 6a), Table 3
gp_vector_problem_large()
#Figure 6b), Table 3
gp_vector_small()
# Figure online appendix, Table 3
benchmark_nsga3_gp_vector_problem_small()
# Figure online appendix, Table 3
benchmark_nsga3_gp_vector_problem_large()

#Figure 4a)
run_time_comparison_GP_Comola()
#Figure 4b)
run_time_comparison_GP_NSGA3()