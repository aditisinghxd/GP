import numpy as np
from help_and_utility_functions.utility_functions import total_area_of_filtered_osgeo_layer, create_new_landuse_layer

def comola_transition_constraint_check(landuse_map,initial_landuse_map, transition_constraints_matrix, landuse_array = None):
     # check if landuse_map contains nan values. If so, make invalid
    if np.count_nonzero(np.isnan(landuse_map)) > 0:
        valid_landuse_map = False
    else:
        valid_landuse_map = True

    # check transition constraints
    # reset every land use back to initial land use if it is constrained
    nr_landuse_classes = transition_constraints_matrix.shape[0] - 1
    for i in range(1, nr_landuse_classes + 1):
        for j in range(1, nr_landuse_classes + 1):
            if transition_constraints_matrix[j, i] == 0:
                transition_back_positions = np.logical_and((initial_landuse_map == j), (landuse_map == i))
                landuse_map[transition_back_positions] = j
                #nr_cells_transition_back = transition_back_positions.sum()
    return valid_landuse_map, landuse_map, None

def comola_area_constraint_check(landuse_map, area_constraints_matrix, landuse_array = None):
    valid_area_constraints_met = True
    nr_landuse_classes = 7
    total_cells = landuse_map.shape[0] * landuse_map.shape[1]
    #nr_landuse_classes = transition_constraints_matrix.shape[0] - 1
    for i in range(1, nr_landuse_classes + 1):
        if not area_constraints_matrix[1][i] <= ((landuse_map == i).sum() / total_cells * 100) <= \
               area_constraints_matrix[2][i]:
            valid_area_constraints_met = False
    return valid_area_constraints_met, landuse_map, None

def vector_problem_permittable_landuses_constraint_check(landuse_map,landuse_array, permitted_landuses):
    valid = True
    nr_valid_units = np.isin(landuse_array[:,1], permitted_landuses).sum()
    if nr_valid_units < landuse_array.shape[0]:
        valid = False
    return valid, landuse_map, landuse_array

def vector_problem_transition_constraint_check(initial_landuse_array,transition_constraints_matrix, landuse_map, landuse_array):
    if np.count_nonzero(np.isnan((landuse_array[:,1]).astype(float))) > 0:
        valid_landuse_map = False
    else:
        valid_landuse_map = True

    # check transition constraints
    # reset every land use back to initial land use if it is constrained
    nr_landuse_classes = transition_constraints_matrix.shape[0] - 1
    for i in range(1, nr_landuse_classes + 1):
        for j in range(1, nr_landuse_classes + 1):
            if transition_constraints_matrix[j, i] == 0:
                transition_back_positions = np.logical_and((initial_landuse_array[:,1].astype(int) == j), (landuse_array[:,1].astype(int) == i))
                landuse_array[transition_back_positions,1] = str(j)

    landuse_map = create_new_landuse_layer(landuse_map, landuse_array[:,1])
    return valid_landuse_map, landuse_map, landuse_array

def vector_problem_area_constraint_check(area_constraints_matrix,landuse_map,landuse_array, total_area):
    area_constraints_met = True
    nr_landuse_classes = 7
    total_cells = landuse_array.shape[0]
    for i in range(1, nr_landuse_classes + 1):
        landuse_coverage = total_area_of_filtered_osgeo_layer(landuse_map.GetLayer(0),filter = "landuse = '{}'".format(i)) / total_area * 100
        if not area_constraints_matrix[1][i] <= landuse_coverage <= \
               area_constraints_matrix[2][i]:
            area_constraints_met = False
        landuse_coverage = total_area_of_filtered_osgeo_layer(landuse_map.GetLayer(0),
                                                              filter="landuse in ('1', '2', '3', '4', '5', '6', '7') ".format(i)) / total_area * 100

    return area_constraints_met, landuse_map, landuse_array