import numpy as np
import os
import functools
import geopandas as geop
from pathlib import Path
import random
from osgeo import ogr
from help_and_utility_functions import plotting
from help_and_utility_functions.CoMOLA_help_functions import read_patch_ID_map, determine_static_classes, create_patch_ID_map
from optimization_setup.objective_functions import compute_habstruct,get_extreme_habitat_heterogeneity, compute_yield, get_extreme_crop_yield,get_extreme_water_yield, compute_sar, get_extreme_species_richness, compute_water_yield,\
compute_distance_living_windplants, compute_urban_neighbors, compute_agriculture_within_water_range, compute_avg_agriculture_unit_size,\
get_extreme_agri_unit_size, get_extreme_wka_distance,get_extreme_agriculture_within_water_range, get_extreme_urban_neighbors
from optimization_setup.constraint_functions import comola_area_constraint_check, comola_transition_constraint_check, \
vector_problem_transition_constraint_check, vector_problem_area_constraint_check, vector_problem_permittable_landuses_constraint_check
from help_and_utility_functions.utility_functions import import_landuse_vector_features, get_polygon_centroids, get_neighbor_matrix, get_distances_from_centroids_to_points,\
    get_minimum_average_distances_from_boundaries_to_waters, get_landuse_array, total_area_of_filtered_osgeo_layer, createBuffer, add_input_data_uncertainty_per_cell

def validate_landuse_map(landuse_map,
                         constraint_functions,
                         landuse_column_name = "landuse"):
    # 1st check: If the landuse_map is of type osgeo.ogr.Layer (vector), then we also want a landuse_array for faster validation and objective computation.
    # In raster and patch case we don't need another landuse_array and set it to None
    if (isinstance(landuse_map, ogr.DataSource)):
        landuse_array = get_landuse_array(landuse_map,landuse_column_name)
    else:
        landuse_array = None

    valid = True
    constraints_met = [valid]

    for constraint in constraint_functions:
        # the other input variables for validating the constraints are defined in the problem section (e.g. initial land use map, area constraint_matrix,...)
        valid, landuse_map, landuse_array = constraint.formula(landuse_map = landuse_map, landuse_array = landuse_array)
        constraints_met.append(valid)
        if valid is False:
            break

    return all(constraints_met), landuse_map, landuse_array

class Objective:
    _id = 0
    def __init__(self,
                 name,
                 minimization,
                 formula,
                 description,
                 extremes):
        self.objective_id = Objective._id
        Objective._id += 1
        self.name = name
        self.minimization = minimization
        self.formula = formula
        self.description = description
        self.extremes = extremes

class Constraint:
    _id = 0
    def __init__(self,
                 name,
                 formula,
                 description):
        self.constraint_id = Constraint._id
        Constraint._id += 1
        self.name = name
        self.formula = formula
        self.description = description

class Problem:
    _id = 0
    def __init__(self,
                 name,
                 objectives,
                 constraints,
                 encoding):
        self.problem_id = Problem._id
        Problem._id += 1
        self.name = name
        self.objectives = objectives
        self.constraints = constraints
        self.encoding = encoding

class LandUseAllocationProblem(Problem):
    def __init__(self,
                 name,
                 objectives,
                 constraints,
                 encoding,                 
                 initial_landuse_map,
                 mapping_points,
                 plot_solution_map,
                 additional_data):
        super().__init__(name,
                 objectives,
                 constraints,
                 encoding)
        self.initial_landuse_map = initial_landuse_map
        self.mapping_points = mapping_points
        self.plot_solution_map = plot_solution_map
        self.additional_data = additional_data

class LandUseAllocationProblemPatches(LandUseAllocationProblem):
    def __init__(self,
                 name,
                 objectives,
                 constraints,
                 encoding,
                 initial_landuse_map,
                 mapping_points,
                 plot_solution_map,
                 patch_map,
                 additional_data):
        super().__init__(name,
                         objectives,
                         constraints,
                         encoding,
                         initial_landuse_map,
                         mapping_points,
                         plot_solution_map,
                         additional_data)
        self.patch_map = patch_map

class LandUseAllocationProblemSingleCells(LandUseAllocationProblem):
    def __init__(self,
                 name,
                 objectives,
                 constraints,
                 encoding,
                 initial_landuse_map,
                 mapping_points,
                 plot_solution_map,
                 additional_data):
        super().__init__(name,
                 objectives,
                 constraints,
                 encoding,
                 initial_landuse_map,
                 mapping_points,
                 plot_solution_map,
                 additional_data)

class LandUseAllocationProblemVector(LandUseAllocationProblem):
    def __init__(self,
                 name,
                 objectives,
                 constraints,
                 encoding,
                 initial_landuse_map,
                 initial_landuse_map_datasource,
                 mapping_points,
                 plot_solution_map,
                 additional_data):
        super().__init__(name,
                         objectives,
                         constraints,
                         encoding,
                         initial_landuse_map,
                         mapping_points,
                         plot_solution_map,
                         additional_data)
        self.initial_landuse_map_datasource = initial_landuse_map_datasource

def comola_single_cells(size = (10, 10)):
    name = "CoMOLA single cells"
    parentdir = Path(__file__).parent
    input_dir = os.path.join(parentdir, '../input_data', 'raster_optimization')

    trans_matrix = np.genfromtxt(os.path.join(input_dir, "transition_matrix.txt"), dtype=int, filling_values='-1')
    initial_landuse_map = np.genfromtxt(os.path.join(input_dir, "initial_landuse.asc"), dtype=float, skip_header=6, delimiter=',')
    area_constraints_matrix = np.genfromtxt(os.path.join(input_dir, "min_max.txt"),
                                            dtype=int, filling_values='-1')

    initial_size = initial_landuse_map.shape

    soilfertility_map = np.empty(size)
    for i in range(soilfertility_map.shape[0]):
        for j in range(soilfertility_map.shape[1]):
            soilfertility_map[i, j] = (1 - (i / soilfertility_map.shape[0]))

    #if new size is smaller just take part of the oroginal initial landuse map
    if size[0] < initial_landuse_map.shape[0]:
        initial_landuse_map = initial_landuse_map[:size[0], :size[1]]

    # if it's the same problem size use the original initial landuse map
    elif size[0] == initial_landuse_map.shape[0]:
        pass

    #if new size is a multiplier of initial landuse map stretch it. works for 20*20, 30*30 etc.
    # in that caase of 30 for example, every cell of the original initial landusemap is repeated 3 times in x and y
    elif size[0]%initial_landuse_map.shape[0] == 0:
        new_initial_landuse_map = np.empty(size, int)
        step_size_reduction = int(new_initial_landuse_map.shape[0] / initial_landuse_map.shape[0])
        for i in range(initial_size[0]):
            for j in range(initial_size[1]):
                for k in range(step_size_reduction):
                    new_initial_landuse_map[(i * step_size_reduction):((i * step_size_reduction) + step_size_reduction),
                    (j * step_size_reduction):((j * step_size_reduction) + step_size_reduction)] = initial_landuse_map[
                        i, j]

        initial_landuse_map = new_initial_landuse_map

    #if new size is larger but no multiplier fill with random value between 1 and 7
    else:
        new_initial_landuse_map = np.empty(size, int)
        new_initial_landuse_map[:initial_landuse_map.shape[0], :initial_landuse_map.shape[1]] = initial_landuse_map
        for i in range(size[0]):
            for j in range(size[1]):
                if i >= initial_landuse_map.shape[0] or j >= initial_landuse_map.shape[1]:
                    new_initial_landuse_map[i, j] = random.randint(1, 8)
        initial_landuse_map = new_initial_landuse_map

    transition_constraint = Constraint(name="Land use transition constraint",
                                       formula=functools.partial(comola_transition_constraint_check,
                                                                 initial_landuse_map=initial_landuse_map,
                                                                 transition_constraints_matrix=trans_matrix),
                                       description=None)

    area_constraint = Constraint(name="Land use area constraint",
                                 formula=functools.partial(comola_area_constraint_check,
                                                           area_constraints_matrix=area_constraints_matrix),
                                 description=None)

    constraints = [transition_constraint, area_constraint]

    compute_extreme_single_objective_solutions = True
    if compute_extreme_single_objective_solutions:
        validation_function = functools.partial(validate_landuse_map, constraint_functions=constraints)
        global_best_CY, global_worst_CY = get_extreme_crop_yield(initial_landuse_map, validation_function,
                                                                 soilfertility_map)
        # global_best_HH, global_worst_HH = get_extreme_habitat_heterogeneity(tmpfile_landuse,
        # initial_landuse_array, neighbor_matrix)

        land_use_forest = 6
        global_best_SR, global_worst_SR = get_extreme_species_richness(initial_landuse_map,
                                                                       max_forest=area_constraints_matrix[2][
                                                                           land_use_forest])
        global_best_HH, global_worst_HH = get_extreme_habitat_heterogeneity(initial_landuse_map, validation_function)
        global_best_WY, global_worst_WY = get_extreme_water_yield(initial_landuse_map, validation_function)
    else:
        global_best_CY, global_worst_CY = None, None
        global_best_SR, global_worst_SR = None, None
        global_best_HH, global_worst_HH = None, None
        global_best_WY, global_worst_WY = None, None

    max_yield_obj = Objective(name="Max. crop yield",
                              minimization=False,
                              formula=functools.partial(compute_yield, soilfertility_map=soilfertility_map),
                              description=None,
                              extremes={'extreme_best': global_best_CY,
                                        'extreme_worst': global_worst_CY})

    max_habstruct_obj = Objective(name="Max. habitat heterogeneity",
                                  minimization=False,
                                  formula=compute_habstruct,
                                  description=None,
                                  extremes={'extreme_best': global_best_HH,
                                            'extreme_worst': global_worst_HH})

    max_sar_obj = Objective(name="Max. forest species richness",
                            minimization=False,
                            formula=compute_sar,
                            description=None,
                            extremes={'extreme_best': global_best_SR,
                                      'extreme_worst': global_worst_SR})

    max_wy_obj = Objective(name="Max. water yield",
                           minimization=False,
                           formula=compute_water_yield,
                           description=None,
                           extremes={'extreme_best': global_best_WY,
                                     'extreme_worst': global_worst_WY})

    objectives = [max_yield_obj, max_habstruct_obj, max_sar_obj, max_wy_obj]


    encoding = "cell"

    mapping_extent = []
    for x in range(initial_landuse_map.shape[0]):
        _X = []
        for y in range(initial_landuse_map.shape[1]):
            _X.append(y)
        mapping_extent.append(_X)

    plot_solution_map = plotting.plot_raster_solution

    return LandUseAllocationProblemSingleCells( name,
                                     objectives,
                                     constraints,
                                     encoding,
                                     initial_landuse_map,
                                     mapping_extent,
                                     plot_solution_map,
                                     additional_data = None)

def comola_single_cells_uncertainty(size = (10, 10), seed = 1):
    name = "CoMOLA single cells uncertainty"
    parentdir = Path(__file__).parent
    input_dir = os.path.join(parentdir, '../input_data', 'raster_optimization')

    trans_matrix = np.genfromtxt(os.path.join(input_dir, "transition_matrix.txt"), dtype=int, filling_values='-1')
    initial_landuse_map = np.genfromtxt(os.path.join(input_dir, "initial_landuse.asc"), dtype=float, skip_header=6, delimiter=',')
    area_constraints_matrix = np.genfromtxt(os.path.join(input_dir, "min_max.txt"),
                                            dtype=int, filling_values='-1')
    error_matrix = np.genfromtxt(os.path.join(input_dir, "land_use_error_matrix.txt"), dtype=float, filling_values='-1')

    initial_size = initial_landuse_map.shape

    soilfertility_map = np.empty(size)
    for i in range(soilfertility_map.shape[0]):
        for j in range(soilfertility_map.shape[1]):
            soilfertility_map[i, j] = (1 - (i / soilfertility_map.shape[0]))

    #if new size is smaller just take part of the oroginal initial landuse map
    if size[0] < initial_landuse_map.shape[0]:
        initial_landuse_map = initial_landuse_map[:size[0], :size[1]]

    # if it's the same problem size use the original initial landuse map
    elif size[0] == initial_landuse_map.shape[0]:
        pass

    #if new size is a multiplier of initial landuse map stretch it. works for 20*20, 30*30 etc.
    # in that caase of 30 for example, every cell of the original initial landusemap is repeated 3 times in x and y
    elif size[0]%initial_landuse_map.shape[0] == 0:
        new_initial_landuse_map = np.empty(size, int)
        step_size_reduction = int(new_initial_landuse_map.shape[0] / initial_landuse_map.shape[0])
        for i in range(initial_size[0]):
            for j in range(initial_size[1]):
                for k in range(step_size_reduction):
                    new_initial_landuse_map[(i * step_size_reduction):((i * step_size_reduction) + step_size_reduction),
                    (j * step_size_reduction):((j * step_size_reduction) + step_size_reduction)] = initial_landuse_map[
                        i, j]

        initial_landuse_map = new_initial_landuse_map

    #if new size is larger but no multiplier fill with random value between 1 and 7
    else:
        new_initial_landuse_map = np.empty(size, int)
        new_initial_landuse_map[:initial_landuse_map.shape[0], :initial_landuse_map.shape[1]] = initial_landuse_map
        for i in range(size[0]):
            for j in range(size[1]):
                if i >= initial_landuse_map.shape[0] or j >= initial_landuse_map.shape[1]:
                    new_initial_landuse_map[i, j] = random.randint(1, 8)
        initial_landuse_map = new_initial_landuse_map

    initial_landuse_map = add_input_data_uncertainty_per_cell(initial_landuse_map, error_matrix, trans_matrix, seed)

    transition_constraint = Constraint(name="Land use transition constraint",
                                       formula=functools.partial(comola_transition_constraint_check,
                                                                 initial_landuse_map=initial_landuse_map,
                                                                 transition_constraints_matrix=trans_matrix),
                                       description=None)

    area_constraint = Constraint(name="Land use area constraint",
                                 formula=functools.partial(comola_area_constraint_check,
                                                           area_constraints_matrix=area_constraints_matrix),
                                 description=None)

    constraints = [transition_constraint, area_constraint]

    compute_extreme_single_objective_solutions = True
    if compute_extreme_single_objective_solutions:
        validation_function = functools.partial(validate_landuse_map, constraint_functions=constraints)
        global_best_CY, global_worst_CY = get_extreme_crop_yield(initial_landuse_map, validation_function,
                                                                 soilfertility_map)
        # global_best_HH, global_worst_HH = get_extreme_habitat_heterogeneity(tmpfile_landuse,
        # initial_landuse_array, neighbor_matrix)

        land_use_forest = 6
        global_best_SR, global_worst_SR = get_extreme_species_richness(initial_landuse_map,
                                                                       max_forest=area_constraints_matrix[2][
                                                                           land_use_forest])
        global_best_HH, global_worst_HH = get_extreme_habitat_heterogeneity(initial_landuse_map, validation_function)
        global_best_WY, global_worst_WY = get_extreme_water_yield(initial_landuse_map, validation_function)
    else:
        global_best_CY, global_worst_CY = None, None
        global_best_SR, global_worst_SR = None, None
        global_best_HH, global_worst_HH = None, None
        global_best_WY, global_worst_WY = None, None

    max_yield_obj = Objective(name="Max. crop yield",
                              minimization=False,
                              formula=functools.partial(compute_yield, soilfertility_map=soilfertility_map),
                              description=None,
                              extremes={'extreme_best': global_best_CY,
                                        'extreme_worst': global_worst_CY})

    max_habstruct_obj = Objective(name="Max. habitat heterogeneity",
                                  minimization=False,
                                  formula=compute_habstruct,
                                  description=None,
                                  extremes={'extreme_best': global_best_HH,
                                            'extreme_worst': global_worst_HH})

    max_sar_obj = Objective(name="Max. forest species richness",
                            minimization=False,
                            formula=compute_sar,
                            description=None,
                            extremes={'extreme_best': global_best_SR,
                                      'extreme_worst': global_worst_SR})

    max_wy_obj = Objective(name="Max. water yield",
                           minimization=False,
                           formula=compute_water_yield,
                           description=None,
                           extremes={'extreme_best': global_best_WY,
                                     'extreme_worst': global_worst_WY})

    objectives = [max_yield_obj, max_habstruct_obj, max_sar_obj, max_wy_obj]


    encoding = "cell"

    mapping_extent = []
    for x in range(initial_landuse_map.shape[0]):
        _X = []
        for y in range(initial_landuse_map.shape[1]):
            _X.append(y)
        mapping_extent.append(_X)

    plot_solution_map = plotting.plot_raster_solution

    return LandUseAllocationProblemSingleCells( name,
                                     objectives,
                                     constraints,
                                     encoding,
                                     initial_landuse_map,
                                     mapping_extent,
                                     plot_solution_map,
                                     additional_data = None)

def comola_patches(size = (10, 10)):
    parentdir = Path(__file__).parent
    input_dir = os.path.join(parentdir, '../input_data', 'raster_optimization')

    trans_matrix = np.genfromtxt(os.path.join(input_dir, "transition_matrix.txt"), dtype=int, filling_values='-1')
    initial_landuse_map = np.genfromtxt(os.path.join(input_dir, "initial_landuse.asc"), dtype=float, skip_header=6,
                                        delimiter=',')
    area_constraints_matrix = np.genfromtxt(os.path.join(input_dir, "min_max.txt"),
                                            dtype=int, filling_values='-1')
    initial_size = initial_landuse_map.shape
    soilfertility_map = np.empty(size)
    for i in range(soilfertility_map.shape[0]):
        for j in range(soilfertility_map.shape[1]):
            soilfertility_map[i, j] = (1 - (i / soilfertility_map.shape[0]))

        # if new size is smaller just take part of the oroginal initial landuse map
        if size[0] < initial_landuse_map.shape[0]:
            initial_landuse_map = initial_landuse_map[:size[0], :size[1]]

        # if it's the same problem size use the original initial landuse map
        elif size[0] == initial_landuse_map.shape[0]:
            pass

        # if new size is a multiplier of initial landuse map stretch it. works for 20*20, 30*30 etc.
        # in that caase of 30 for example, every cell of the original initial landusemap is repeated 3 times in x and y
        elif size[0] % initial_landuse_map.shape[0] == 0:
            new_initial_landuse_map = np.empty(size, int)
            step_size_reduction = int(new_initial_landuse_map.shape[0] / initial_landuse_map.shape[0])
            for i in range(initial_size[0]):
                for j in range(initial_size[1]):
                    for k in range(step_size_reduction):
                        new_initial_landuse_map[
                        (i * step_size_reduction):((i * step_size_reduction) + step_size_reduction),
                        (j * step_size_reduction):((j * step_size_reduction) + step_size_reduction)] = \
                        initial_landuse_map[
                            i, j]

            initial_landuse_map = new_initial_landuse_map

        # if new size is larger but no multiplier fill with random value between 1 and 7
        else:
            new_initial_landuse_map = np.empty(size, int)
            new_initial_landuse_map[:initial_landuse_map.shape[0], :initial_landuse_map.shape[1]] = initial_landuse_map
            for i in range(size[0]):
                for j in range(size[1]):
                    if i >= initial_landuse_map.shape[0] or j >= initial_landuse_map.shape[1]:
                        new_initial_landuse_map[i, j] = random.randint(1, 8)
            initial_landuse_map = new_initial_landuse_map

    max_range = 8
    four_neighbours = "False"
    static_elements, nonstatic_elements = determine_static_classes(trans_matrix, max_range)

    new_patch_map = create_patch_ID_map(initial_landuse_map, str(-2), static_elements, four_neighbours)
    patch_ID_map, encoded_initial_solution = read_patch_ID_map(new_patch_map,
                                                               initial_landuse_map, 0, static_elements, four_neighbours)
    mapping_extent = [x for x in range(initial_landuse_map.shape[0])]

    transition_constraint = Constraint(name="Land use transition constraint",
                                       formula = functools.partial(comola_transition_constraint_check,
                                                                   initial_landuse_map=initial_landuse_map,
                                                                   transition_constraints_matrix = trans_matrix),
                                       description=None)

    area_constraint = Constraint(  name = "Land use area constraint",
                                         formula = functools.partial(comola_area_constraint_check, area_constraints_matrix = area_constraints_matrix),
                                         description = None)

    constraints = [transition_constraint, area_constraint]

    compute_extreme_single_objective_solutions = True
    if compute_extreme_single_objective_solutions:
        validation_function = functools.partial(validate_landuse_map, constraint_functions=constraints)
        global_best_CY, global_worst_CY = get_extreme_crop_yield(initial_landuse_map, validation_function, soilfertility_map)
        # global_best_HH, global_worst_HH = get_extreme_habitat_heterogeneity(tmpfile_landuse,
        # initial_landuse_array, neighbor_matrix)

        land_use_forest = 6
        global_best_SR, global_worst_SR = get_extreme_species_richness(initial_landuse_map, max_forest= area_constraints_matrix[2][land_use_forest])
        global_best_HH, global_worst_HH = get_extreme_habitat_heterogeneity(initial_landuse_map, validation_function)
        global_best_WY, global_worst_WY = get_extreme_water_yield(initial_landuse_map,validation_function)
    else:
        global_best_CY, global_worst_CY = None, None
        global_best_SR, global_worst_SR = None, None
        global_best_HH, global_worst_HH = None, None
        global_best_WY, global_worst_WY = None, None


    max_yield_obj = Objective(name="Max. crop yield",
                              minimization=False,
                              formula=functools.partial(compute_yield, soilfertility_map=soilfertility_map),
                              description=None,
                              extremes={'extreme_best': global_best_CY,
                                        'extreme_worst': global_worst_CY})

    max_habstruct_obj = Objective(name="Max. habitat heterogeneity",
                                  minimization=False,
                                  formula=compute_habstruct,
                                  description=None,
                                  extremes={'extreme_best': global_best_HH,
                                            'extreme_worst': global_worst_HH})

    max_sar_obj = Objective(name="Max. forest species richness",
                            minimization=False,
                            formula=compute_sar,
                            description=None,
                            extremes={'extreme_best': global_best_SR,
                                      'extreme_worst': global_worst_SR})

    max_wy_obj = Objective(name="Max. water yield",
                           minimization=False,
                           formula=compute_water_yield,
                           description=None,
                           extremes={'extreme_best': global_best_WY,
                                     'extreme_worst': global_worst_WY})

    objectives = [max_yield_obj, max_habstruct_obj, max_sar_obj, max_wy_obj]


    encoding = "patch"

    plot_solution_map = plotting.plot_raster_solution

    return LandUseAllocationProblemPatches(name = "CoMOLA with land use patches",
    objectives = objectives,
    constraints = constraints,
    encoding = encoding,
    initial_landuse_map = initial_landuse_map,
    patch_map = patch_ID_map,
    mapping_points = mapping_extent,
    plot_solution_map = plot_solution_map,
    additional_data = None)

def test_problem_vector(study_area = '100_fluren'):
    parentdir = Path(__file__).parent
    input_dir = os.path.join(parentdir, '../input_data', 'vector_optimization', study_area)
    trans_matrix = np.genfromtxt(os.path.join(parentdir, '../input_data', 'vector_optimization', "transition_matrix.txt"), dtype=int,
                                 filling_values='-1')
    area_constraints_matrix = np.genfromtxt(os.path.join(parentdir, '../input_data', 'vector_optimization', "min_max.txt"),
                                            dtype=int, filling_values='-1')
    #define the landuse column name of the input landuse file
    landuse_column_name = "landuse_re"
    tmpfile_landuse = import_landuse_vector_features(os.path.join(input_dir, "landnutzung.shp"),
                                                                  mem_layer_name = "landuse_tmp")
    layer_landuse = tmpfile_landuse.GetLayer(0)

    tmpfile_water = import_landuse_vector_features(os.path.join(input_dir, "fliessgewaesser.shp"),
                                                     mem_layer_name="water_tmp")
    tmpfilewaterbuffer = createBuffer(tmpfile_water, os.path.join(input_dir, "waters_buffer_500m.shp"), 500)

    # create network representation for being able to identify neighbors and contiguous units
    gdf_landuses = geop.read_file(os.path.join(input_dir, "landnutzung.shp"))
    gdf_wka = geop.read_file(os.path.join(input_dir, "wka.shp"))
    gdf_waters = geop.read_file(os.path.join(input_dir, "fliessgewaesser.shp"))

    neighbor_matrix = get_neighbor_matrix(input_dir, gdf_landuses)

    distance_matrix_points = get_distances_from_centroids_to_points(gdf_landuses, gdf_wka, input_dir)
    initial_landuse_array = get_landuse_array(tmpfile_landuse, landuse_column_name = landuse_column_name)

    print("The selected data format for the land use is vector data. "
          "There are multiple choices how the continuous maps from the Genetic Programming functions "
          "are mapped to the vector representation land use maps with the value of the polygon centroids.")
    print('EPSG from input vector layer: "%s"' %
          layer_landuse.GetSpatialRef().ExportToWkt().rsplit('"EPSG","', 1)[-1].split('"')[0])
    x_min, x_max, y_min, y_max = layer_landuse.GetExtent()
    print("The extent is  " + str(x_min) + ", " + str(x_max) + ", " + str(y_min) + ", " + str(y_max))
    # get the centroids of the polygons and use those for the mapping
    mapping_extent = get_polygon_centroids(layer_landuse)
    total_area = total_area_of_filtered_osgeo_layer(layer_landuse,filter = "{} in ('1','2','3','4','5','6','7') ".format(landuse_column_name))

    permittable_landuses_constraint = Constraint(name= "Permittable landuse constraint",
                                      formula=functools.partial(vector_problem_permittable_landuses_constraint_check,
                                        permitted_landuses=[str(i) for i in range(1,8)]),
                                      description=None)

    transition_constraint = Constraint(name="Land use transition constraint",
                                       formula=functools.partial(vector_problem_transition_constraint_check,
                                                                 initial_landuse_array= initial_landuse_array,
                                                                 transition_constraints_matrix=trans_matrix),
                                       description=None)

    area_constraint = Constraint(name = "Land use area constraint",
                                         formula = functools.partial(vector_problem_area_constraint_check,
                                         area_constraints_matrix = area_constraints_matrix, total_area = total_area),
                                         description = None)

    constraints = [permittable_landuses_constraint,transition_constraint, area_constraint]

    compute_extreme_single_objective_solutions = True
    if compute_extreme_single_objective_solutions:
        global_best_agri_unit_size, global_worst_agri_unit_size = get_extreme_agri_unit_size(tmpfile_landuse,
                                                                                             initial_landuse_array)
        global_best_max_urban_neighbors, global_worst_max_urban_neighbors = get_extreme_urban_neighbors(tmpfile_landuse,
        initial_landuse_array, neighbor_matrix)

        global_best_agriculture_within_water_range, global_worst_agriculture_within_water_range = get_extreme_agriculture_within_water_range(
            tmpfile_landuse,initial_landuse_array, tmpfilewaterbuffer)

        landuse_living = 6
        global_best_wka_distance, global_worst_wka_distance = get_extreme_wka_distance(tmpfile_landuse,initial_landuse_array,
                                                                                       distance_matrix_points, # change farthest units to living (id = 6) until min_percent is reached
                                                                                       landuse = str(landuse_living), min_percent = area_constraints_matrix[1][6], constrained_landuses = trans_matrix[trans_matrix[:, landuse_living] == 0][:,0])
    else:
        global_best_max_urban_neighbors, global_worst_max_urban_neighbors = None, None
        global_best_agriculture_within_water_range, global_worst_agriculture_within_water_range = None, None
        global_best_agri_unit_size, global_worst_agri_unit_size = None, None
        global_best_wka_distance, global_worst_wka_distance = None, None

    max_urban_neighbors_obj = Objective(name="Max. urban neighbors count",
                                        minimization=False,
                                        formula=functools.partial(compute_urban_neighbors,
                                                                  neighbor_matrix=neighbor_matrix),
                                        description=None,
                                        extremes={'extreme_best': global_best_max_urban_neighbors,
                                                  'extreme_worst': global_worst_max_urban_neighbors})

    max_agriculture_within_water_range_obj = Objective(name="Max. agriculture in water range (ha)",
                                                       minimization=False,
                                                       formula=functools.partial(compute_agriculture_within_water_range,
                                                                                 waters_buffer=tmpfilewaterbuffer),
                                                       description=None,
                                                       extremes={
                                                           'extreme_best': global_best_agriculture_within_water_range,
                                                           'extreme_worst': global_worst_agriculture_within_water_range})

    max_agri_unit_size_obj = Objective(name="Max. avg. agriculture size (ha)",
                                       minimization=False,
                                       formula=compute_avg_agriculture_unit_size,
                                       description=None,
                                       extremes={'extreme_best': global_best_agri_unit_size,
                                                 'extreme_worst': global_worst_agri_unit_size})

    max_distance_living_windplants_obj = Objective(name="Max. dist. living to windplants (km)",
                                                   minimization=False,
                                                   formula=functools.partial(compute_distance_living_windplants,
                                                                             distance_matrix_points=distance_matrix_points),
                                                   description=None,
                                                   extremes={'extreme_best': global_best_wka_distance,
                                                             'extreme_worst': global_worst_wka_distance})

    objectives = [max_urban_neighbors_obj, max_agri_unit_size_obj, max_distance_living_windplants_obj,
                  max_agriculture_within_water_range_obj]

    plot_solution = plotting.plot_vector_solution

    return LandUseAllocationProblemVector("Vector land use optimization",
                 objectives = objectives,
                 constraints = constraints,
                 encoding = "vector",
                 initial_landuse_map = layer_landuse,
                 initial_landuse_map_datasource = tmpfile_landuse,
                 mapping_points = mapping_extent,
                 plot_solution_map= plot_solution,
                 additional_data = {"geodataframe_waters": gdf_waters, "geodataframe_wka": gdf_wka })