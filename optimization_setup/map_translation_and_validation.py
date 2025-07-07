import numpy as np
from osgeo import ogr
from help_and_utility_functions.utility_functions import create_new_landuse_layer, get_landuse_array

def continuous_map_to_landuse_classes(continuous_map, landuse_mapping, landuse_range = 7):
    np.seterr(invalid='ignore')
    #normalize to range 1- landuse_range
    max_value = np.max(continuous_map)
    min_value = np.min(continuous_map)
    normalized_values = np.round((((max_value - continuous_map) / (max_value - min_value)) * (landuse_range - 1)) + 1,0).astype(int)
    c_normalized_values = np.empty(continuous_map.shape, int)
    if landuse_mapping is not None:
        #now apply the mapping
        for i, luc in enumerate(landuse_mapping):
            replace = normalized_values == luc
            c_normalized_values[replace] = i + 1
    return c_normalized_values

def continuous_map_to_landuse_patches(continuous_map, landuse_mapping, patch_map, landuse_range = 7):
    np.seterr(invalid='ignore')
    max_patch_id = patch_map.max()
    for i in range(1,max_patch_id+1):
        continuous_map[patch_map==i] = np.mean(continuous_map[patch_map==i])

    max_value = np.max(continuous_map)
    min_value = np.min(continuous_map)
    #normalize the continuous values to land use order range
    normalized_values = np.round((((max_value - continuous_map) / (max_value - min_value)) * (landuse_range - 1)) + 1,0).astype(int)
    c_normalized_values = np.empty(continuous_map.shape,int)
    if landuse_mapping is not None:
        # now apply the mapping
        for i, luc in enumerate(landuse_mapping):
            replace = normalized_values == luc
            c_normalized_values[replace] = i + 1
    return c_normalized_values

def continuous_map_to_landuse_vector_patches(continuous_map, landuse_mapping, vector_layer, landuse_range = 7):
    np.seterr(invalid='ignore')
    max_value = np.max(continuous_map)
    min_value = np.min(continuous_map)
    normalized_values = np.round((((max_value - continuous_map) / (max_value - min_value)) * (landuse_range - 1)) + 1, 0).astype(int)
    c_normalized_values = np.empty(continuous_map.shape, int)
    if landuse_mapping is not None:
        # now apply the mapping
        for i, luc in enumerate(landuse_mapping):
            replace = normalized_values == luc
            c_normalized_values[replace] = i + 1
    new_landuse_layer = create_new_landuse_layer(vector_layer,c_normalized_values)

    return new_landuse_layer, c_normalized_values

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

def convert_function_to_map(func, points, landuse_problem, landuse_mapping = None):
    landuse_encoding = landuse_problem.encoding
    landuse_array = None

    if landuse_encoding == "cell":
        representation_from_function = np.zeros([len(points), len(points[0])])
        for x in range(len(points)):
            for y in range(len(points[0])):
                representation_from_function[x, y] = func(x, y)
        landuse_map = continuous_map_to_landuse_classes(representation_from_function, landuse_mapping, landuse_range = landuse_mapping.shape[0])

    elif landuse_encoding == "patch":
        representation_from_function = np.zeros([len(points),len(points)])
        patch_map = landuse_problem.patch_map
        for x in points:
            for y in points:
                representation_from_function[x, y] = func(x, y)
        landuse_map = continuous_map_to_landuse_patches(representation_from_function, landuse_mapping, patch_map, landuse_range = landuse_mapping.shape[0])


    elif landuse_encoding == "vector":

        representation_from_function = np.zeros([len(points), 1])

        for i in range(len(points)):
            representation_from_function[i] = func(points[i][0], points[i][1])

        landuse_map, land_uses = continuous_map_to_landuse_vector_patches(representation_from_function, landuse_mapping,
                                                                          landuse_problem.initial_landuse_map_datasource,
                                                                          landuse_range=landuse_mapping.shape[0])

        # plot_vector_layer(landuse_map,land_uses)

    else:

        print("select 'cell', 'patch' or 'vector' for landuse_encocoding")

    return landuse_map
