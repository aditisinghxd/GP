from deap import creator
from deap import gp
from deap import base
from deap import tools
import operator
import math
import random
import numpy as np
from .gp_adaptions import initialize_with_landuse_order, cxOnePoint_adapted, mutUniform_adapted
from .map_translation_and_validation import validate_landuse_map, convert_function_to_map

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

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def define_lu_class_order(a):
    return random.randint(0, 100)

def get_deap_gp_specifications(landuse_problem, evolution_strategy = "nsga3"):

    def evalSymbReg(individual, points):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)

        # convert function into landuse map. If vector optimization a landuse_array(numpy) is also returned, otherwise it is None
        landuse_map = convert_function_to_map(func, points, landuse_problem, individual.land_use_order)

        # validate landuse map. If valid, go to evaluation
        validity, landuse_map, landuse_array = validate_landuse_map(landuse_map, landuse_problem.constraints)

        if validity:
            # evaluate solution for all objectives
            objective_values = [obj.formula(landuse_map=landuse_map, landuse_array=landuse_array) for obj in
                                landuse_problem.objectives]
        else:
            objective_values = tuple([float("inf") * -1 if obj.minimization is False else float("inf") for obj in
                                      landuse_problem.objectives])
        return objective_values

    # get the objective weights. If objective is to be maximized set 1.0, otherwiese -1.0
    objective_weights = [1.0 if o.minimization is False else -1.0 for o in landuse_problem.objectives]
    creator.create("FitnessMulti", base.Fitness, weights=(objective_weights))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    pset = gp.PrimitiveSet("MAIN", 2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.renameArguments(ARG0='x', ARG1='y')


    if evolution_strategy == "ea_simple":
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", initialize_with_landuse_order, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", cxOnePoint_adapted)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", mutUniform_adapted, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
        toolbox.register("evaluate", evalSymbReg, points=landuse_problem.mapping_points)

    elif evolution_strategy == "nsga3":
        ref_points = uniform_reference_points(nobj=len(landuse_problem.objectives), p=6, max_obj_values = [obj.extremes['extreme_best'] for obj in landuse_problem.objectives if obj.extremes is not None] )
        #ref_points = tools.uniform_reference_points(len(landuse_problem.objectives), 8)
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", initialize_with_landuse_order, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points, return_memory=True)
        toolbox.register("mate", cxOnePoint_adapted)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", mutUniform_adapted, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
        toolbox.register("evaluate", evalSymbReg, points=landuse_problem.mapping_points)

    landuse_problem.toolbox = toolbox
    landuse_problem.pset = pset
    landuse_problem.creator = creator