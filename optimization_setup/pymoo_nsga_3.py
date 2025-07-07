from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from help_and_utility_functions.utility_functions import get_feature_count, create_new_landuse_layer
from optimization_setup.map_translation_and_validation import  validate_landuse_map
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
import numpy as np

def uniform_reference_points(nobj, p=4, scaling=None, max_obj_values = None):
    """Generate reference points uniformly on the hyperplane intersecting
    each axis at 1. The scaling factor is used to combine multiple layers of
    reference points.
    """
    def gen_refs_recursive(ref, nobj, left, total, depth):
        points = []
        if depth == nobj - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                points.extend(gen_refs_recursive(ref.copy(), nobj, left - i, total, depth + 1))
        return points

    ref_points = np.array(gen_refs_recursive(np.zeros(nobj), nobj, p, p, 0))
    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / nobj

    if max_obj_values is not None:
        for i, max in enumerate(max_obj_values):
            ref_points[:,i] = ref_points[:,i] * max
    return ref_points

class BinaryRandomSpatialSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        #make random function to get land use between 0 and 7
        land_use_arrays = np.random.choice([i for i in range(1,8)], size = [n_samples, problem.n_var])
        return land_use_arrays

class LanduseProblemPymoo(Problem):

    # by calling the super() function the problem properties are initialized
    def __init__(self, landuse_problem):
        super().__init__(landuse_problem = landuse_problem)
        self.n_var = get_feature_count(landuse_problem.initial_landuse_map)  # nr of variables
        self.n_obj = len(landuse_problem.objectives)  # nr of objectives
        self.xl = np.array([1 for i in range(self.n_var)])  # lower boundaries
        self.xu = np.array([7 for i in range(self.n_var)])
        self.n_eq_constr=1# upper boundaries
    # the _evaluate function needs to be overwritten from the superclass
    # the method takes two-dimensional NumPy array x with n rows and n columns as input
    # each row represents an individual and each column an optimization variable
    def _evaluate(self, X, out, *args, **kwargs):
        all_obj_values = []
        constraint_handling = []
        for individual in X:
            #landuse_map = create_new_landuse_layer(self.data["landuse_problem"].initial_landuse_map_datasource, individual.astype(str))
            landuse_problem = self.data["landuse_problem"]

            if hasattr(landuse_problem, "initial_landuse_map_datasource"):
                # vector-based case
                landuse_map = create_new_landuse_layer(landuse_problem.initial_landuse_map_datasource, individual.astype(str))
            else:
                # raster-based case (use numpy directly)
                landuse_map = individual.astype(int).reshape(landuse_problem.initial_landuse_map.shape)

            # validate landuse map. If valid, go to evaluation
            validity, landuse_map, landuse_array = validate_landuse_map(landuse_map, self.data["landuse_problem"].constraints)

            if validity:
                # evaluate solution for all objectives
                objective_values = [obj.formula(landuse_map=landuse_map, landuse_array=landuse_array) * -1 for obj in
                                    self.data["landuse_problem"].objectives]
                constraint_handling.append(0.)

            else:
                objective_values = [float("inf")  if obj.minimization is False else float("inf") * -1 for obj in
                                          self.data["landuse_problem"].objectives]
                constraint_handling.append(1.)

            all_obj_values.append(objective_values)
        # after doing the necessary calculations,
        # the objective values have to be added to the dictionary out
        # with the key F and the constrains with key G
        out["F"] = np.array(all_obj_values)
        out["H"] = np.array(constraint_handling)


def nsga3(landuse_problem, seed, n_gen, pop_size):
    # create the reference directions to be used for the optimization
    ref_dirs = uniform_reference_points(nobj=len(landuse_problem.data["landuse_problem"].objectives), p=6,
                                          max_obj_values=[obj.extremes['extreme_best'] for obj in
                                                          landuse_problem.data["landuse_problem"].objectives if obj.extremes is not None])

    sampling = BinaryRandomSpatialSampling()

    algorithm = NSGA3(pop_size=pop_size,
                      ref_dirs=ref_dirs,
                      crossover=SBX(prob=.8, eta=3.0, vtype=float, repair=RoundingRepair()),
                      mutation=PM(prob=.1, eta=3.0, vtype=float, repair=RoundingRepair()),
                      sampling = sampling)

    # execute the optimization
    res = minimize(landuse_problem,
                   algorithm,
                   #save_history = True,
                   seed=seed,
                   termination=('n_gen', n_gen),
                   verbose=True
                   )

    final_population = [{'fitness': i.F, 'representation': i.X} for i in res.pop]
    pf = {'fitness': [i.F for i in res.opt], 'representation': [i.F for i in res.opt]}
    return {"population": final_population, "pareto_front": pf }
