import random
import pickle
import time
from optimization_setup.problem import test_problem_vector,  comola_single_cells, comola_patches, comola_single_cells_uncertainty
from optimization_setup.pymoo_nsga_3 import nsga3, LanduseProblemPymoo


def land_use_allocation_ga(landuse_problem, algorithm, ngen, npop, inner_optimization_loops = 1):
    start = time.time()

    optimization_logger = {"nr_generations": ngen, "population_size": npop, "runs": {}}

    # random state for optimization
    for j in range(inner_optimization_loops):
        # set random state for inner loop with the same land use order
        random.seed(j)

        #translate our land use problem to Pymoo class
        landuse_problem_pymoo = LanduseProblemPymoo(landuse_problem)

        res = algorithm(landuse_problem_pymoo,
                       seed=random.seed(j),
                       n_gen = ngen,
                       pop_size = npop,

                        )
        optimization_logger["runs"][j] = res
    end = time.time()
    print("Optimization finished after " + str(int((end - start) / 60)) + " mins with " + str(inner_optimization_loops) + " inner loops.")
    return optimization_logger


def main():
    p_size = '100_fluren'
    #p_size = 10
    landuse_problem = test_problem_vector(p_size)
    #landuse_problem = comola_patches((p_size,p_size))

    # define on how many times the optimization shall be executed with different random states
    inner_optimization_loops = 1

    ngen = 100
    npop = 200

    run_intrinsic_landuse_order = land_use_allocation_ga(landuse_problem,nsga3, ngen, npop,
                                                            inner_optimization_loops)

    with open(r'output_data\run_time_analysis\ga_{}_nsga3_{}_gens_{}_p_size{}.pkl'.format(landuse_problem.encoding, ngen, npop, p_size),
              'wb') as handle:
        pickle.dump(run_intrinsic_landuse_order, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_time_analysis(problem_type = "raster"):
    #psizes = ['10_fluren', '20_fluren','30_fluren', '40_fluren', '50_fluren','60_fluren', '70_fluren', '80_fluren', '90_fluren', '100_fluren']
    if problem_type == "raster":
        psizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ,110 , 120, 130, 140, 150]
    elif problem_type == "vector":
        #psizes = ['10_fluren', '20_fluren','30_fluren', '40_fluren', '50_fluren','60_fluren', '70_fluren', '80_fluren', '90_fluren', '100_fluren']
        psizes = ['50_fluren']
    else:
        print("Select problem type vector or raster")
        
    for p_size in psizes:
        landuse_problem = None
        #problem_type = "vector"
        #landuse_problem = test_problem_vector(p_size)

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
        run_intrinsic_landuse_order = land_use_allocation_ga(landuse_problem, nsga3, ngen, npop,
                                                             inner_optimization_loops)

        with open(r'output_data\run_time_analysis\gp_{}_nsga3_{}_gens_{}_popsize_problem_size{}.pkl'.format(
                landuse_problem.encoding, ngen, npop, problem_name), 'wb') as handle:
            pickle.dump(run_intrinsic_landuse_order, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # with open(r'output_data\ga_vector_nsga3_10_gens_10_popsize.pkl', 'rb') as handle:
    #     pf_intrinsic_land_use_order_vector = pickle.load(handle)
    #main()
    run_time_analysis(problem_type="vector")
    #run_time_analysis(problem_type="raster")