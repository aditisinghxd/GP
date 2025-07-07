import copy

import numpy as np
from help_and_utility_functions.utility_functions import get_avg_area_of_dissolved_filtered, get_intersection_area, create_new_landuse_layer, iteratively_fill_farthest_units

def compute_water_yield(landuse_map, landuse_array = None):
    # Kc coefficients
    # for landuses 1 2 3 4 5 6 7
    Kc = [0.9, 0.925, 0.95, 0.975, 1, 0.96, 1.14]
    # count land use classes except urban
    landuseclasscounts = []
    for i in range(1, 8):
        landuseclasscounts.append((landuse_map == i).sum())
    # calculate water yield as sum of inverted Kc values
    return np.sum([landuseclasscounts[i] * (1 / Kc[i]) for i in range(len(landuseclasscounts))])

def get_extreme_water_yield(initial_landuse_map,validation_function):
    max_wyld_solution = np.full(initial_landuse_map.shape, 1., dtype=float)
    valid, validadet_max_wyld_landusemap, empty = validation_function(landuse_map= max_wyld_solution)
    wyld_max = compute_water_yield(validadet_max_wyld_landusemap)
    return wyld_max, 0


def compute_yield(landuse_map, soilfertility_map, landuse_array = None):
    ##########################################################################################
    #    land_use.asc        |land use map containing the following classes
    #                        |1,2,3,4,5 = arable land with increasing intensity from 1 to 5
    #                        |6 = forest
    #                        |7 = pasture
    #                        |8 = urban area
    #                        |-2 = no data
    #    soil_fertility.asc  |map on soil fertility which can range from 0.1 to 1
    ##########################################################################################
    arable_index = landuse_map <= 5
    yieldmap = np.nan_to_num(np.log(landuse_map[arable_index] * (1 + soilfertility_map[arable_index])))
    total_yield = np.sum(yieldmap)
    return total_yield

def get_extreme_crop_yield(initial_landuse_map, validation_function, soilfertility_map):
    maxyield_solution = np.full(initial_landuse_map.shape, 5., dtype=float)
    valid, validadet_max_yield_landusemap, empty = validation_function(landuse_map = maxyield_solution)
    cropyield_max = compute_yield(validadet_max_yield_landusemap, soilfertility_map)
    return cropyield_max, 0

def compute_sar(landuse_map, landuse_array = None):
    forest_area = (landuse_map== 6).sum()
    # calculate species richness based on species-area relationship (formula: S=c*A^z)
    c = 5
    z = 0.2
    S = c * forest_area ** z
    return S

def get_extreme_species_richness(initial_landuse_map, max_forest):
    sar_max = 5 * ((initial_landuse_map.shape[0] * initial_landuse_map.shape[1]) * (max_forest/100)) ** 0.2
    return sar_max,0

def compute_habstruct(landuse_map, landuse_array = None):
    def edgecount(recoded_landusemap):
        # R code pseudo-code:
        # 1. create two matrices. One for row-wise counting and one for column-wise counting
        # 2. For both the row- and column-wise edge counting: compare neighbouring cells. If neighbouring cell not the same: Fill True
        rowcomparison_copy = recoded_landusemap.copy()
        colcomparison_copy = recoded_landusemap.copy()
        nr_rows = recoded_landusemap.shape[0]
        # insert row and compare
        neighborrow_edge_counter = 0
        neighborcol_edge_counter = 0
        for i in range(recoded_landusemap.shape[0] - 1):
            for j in range(recoded_landusemap.shape[1] - 1):
                if recoded_landusemap[i, j] != recoded_landusemap[i + 1, j] \
                        and recoded_landusemap[i, j] != np.nan \
                        and recoded_landusemap[i + 1, j] != np.nan:
                    neighborrow_edge_counter += 1
                    rowcomparison_copy[i, j] = True
                if recoded_landusemap[i, j] != recoded_landusemap[i, j + 1] \
                        and recoded_landusemap[i, j] != np.nan \
                        and recoded_landusemap[i, j + 1] != np.nan:
                    neighborcol_edge_counter += 1
                    colcomparison_copy[i, j] = True
        # last row
        for j in range(recoded_landusemap.shape[1] - 1):
            if recoded_landusemap[-1, j] != recoded_landusemap[-1, j + 1] \
                    and recoded_landusemap[-1, j] != np.nan \
                    and recoded_landusemap[-1, j + 1] != np.nan:
                neighborcol_edge_counter += 1
                colcomparison_copy[-1, j] = True
        # last col
        for i in range(recoded_landusemap.shape[0] - 1):
            if recoded_landusemap[i, -1] != recoded_landusemap[i + 1, -1] \
                    and recoded_landusemap[i, -1] != np.nan \
                    and recoded_landusemap[i + 1, -1] != np.nan:
                neighborrow_edge_counter += 1
                colcomparison_copy[-1, j] = True
        return neighborcol_edge_counter + neighborrow_edge_counter

    def recode_landuse_map(landusemap, recode_mapping):
        # Recode land use map so that edges between not considered land use classes are ignored
        # replace cells with arable land
        lumap_copy = landusemap.copy()
        for i in recode_mapping:
            try:
                lumap_copy[lumap_copy == i] = recode_mapping[i]
            except:
                pass
        return lumap_copy

    ####################  Count only "full" edges (=1)  ####################
    # Recode land use map so that edges between not considered land use classes are ignored
    recode_mapping = {
        2: -2,
        3: -2,
        4: -2,
        5: -2,
        8: -2,
        -2: None
    }

    recoded_landusemap = recode_landuse_map(landuse_map, recode_mapping)
    nr_full_edges = edgecount(recoded_landusemap)

    ####################  Count only edges with arable with land with intensity = 2 (=0.5)  ####################
    # Recode land use map so that edges between not considered land use classes are ignored
    recode_mapping_weighted = {
        1: -2,
        3: -2,
        4: -2,
        5: -2,
        8: -2,
        6: 1,
        7: 1,
        -2: None
    }
    recoded_landusemap_weighted = recode_landuse_map(landuse_map, recode_mapping_weighted)
    nr_weighted_edges = float(edgecount(recoded_landusemap_weighted)) / 2

    ####################  Count only edges with arable with land with intensity = 3 (=1/3)  ####################
    # Recode land use map so that edges between not considered land use classes are ignored
    recode_mapping_weighted2 = {
        1: -2,
        2: -2,
        4: -2,
        5: -2,
        8: -2,
        6: 1,
        7: 1,
        -2: None
    }
    recoded_landusemap_weighted2 = recode_landuse_map(landuse_map, recode_mapping_weighted2)
    nr_weighted_edges2 = float(edgecount(recoded_landusemap_weighted2)) / 3

    ####################  Count only edges with arable with land with intensity = 4 (=1/4)  ####################
    # Recode land use map so that edges between not considered land use classes are ignored
    recode_mapping_weighted3 = {
        1: -2,
        2: -2,
        3: -2,
        5: -2,
        8: -2,
        6: 1,
        7: 1,
        -2: None
    }
    recoded_landusemap_weighted3 = recode_landuse_map(landuse_map, recode_mapping_weighted3)
    nr_weighted_edges3 = float(edgecount(recoded_landusemap_weighted3)) / 4

    ####################  Count only edges with arable with land with intensity = 5 (=1/5)  ####################
    # Recode land use map so that edges between not considered land use classes are ignored
    recode_mapping_weighted4 = {
        1: -2,
        2: -2,
        3: -2,
        4: -2,
        8: -2,
        6: 1,
        7: 1,
        -2: None
    }

    recoded_landusemap_weighted4 = recode_landuse_map(landuse_map, recode_mapping_weighted4)
    nr_weighted_edges4 = float(edgecount(recoded_landusemap_weighted4)) / 5

    return nr_full_edges + nr_weighted_edges + nr_weighted_edges2 + nr_weighted_edges3 + nr_weighted_edges4

def get_extreme_habitat_heterogeneity(initial_landuse_map, validation_function):
    # # make chessboard pattern with cropland 1 and 2
    max_habstruct_map = np.full(initial_landuse_map.shape, 1., dtype=float)
    for i in range(max_habstruct_map.shape[0]):
        for j in range(max_habstruct_map.shape[1]):
            if i%2 == 0:
                if j%2 == 0:
                    max_habstruct_map[i,j] = 2
            else:
                if j%2 != 0:
                    max_habstruct_map[i, j] = 2
    # validate
    valid, validadet_max_habstruct_landusemap, empty = validation_function(landuse_map = max_habstruct_map)
    # add missing forests in field 2 until area constraint of forest is met
    nr_forest_cells = (validadet_max_habstruct_landusemap == 6).sum()
    nr_open_forest_cells = (initial_landuse_map.shape[0] * initial_landuse_map.shape[0] * 0.25) - nr_forest_cells
    for i in range(validadet_max_habstruct_landusemap.shape[0]):
        for j in range(validadet_max_habstruct_landusemap.shape[1]):
            if nr_open_forest_cells >= 1 and validadet_max_habstruct_landusemap[i,j] == 2:
                validadet_max_habstruct_landusemap[i, j] = 6
                nr_open_forest_cells -= 1
    valid, validadet_max_habstruct_landusemap, empty = validation_function(landuse_map=validadet_max_habstruct_landusemap)
    habstruct_max = compute_habstruct(validadet_max_habstruct_landusemap)
    return habstruct_max,0


def compute_distance_living_windplants(landuse_map, landuse_array, distance_matrix_points):
    # identify all landuse_array living (6)
    living = np.where(landuse_array[:, 1] == '6')
    living_ids = landuse_array[living, 0].T
    distance_matrix_selection = distance_matrix_points[np.isin(distance_matrix_points['id'].values, living_ids)]
    #get average total distance for multipolygons
    grouped_distances = distance_matrix_selection.groupby(['id']).mean()
    #return mean distance in km
    mean_distance = grouped_distances['distance'].mean() / 1000
    return mean_distance

def get_extreme_wka_distance(landuse_map, landuse_array, distance_matrix_points, landuse, min_percent = 10, constrained_landuses = ['1', '5', '7' ]):
    #distance matrix benutzen bis minimum living constraint erfüllt ist
    max_distance = iteratively_fill_farthest_units(landuse_map, distance_matrix_points, landuse = landuse, min_percent = min_percent, constrained_landuses = constrained_landuses)
    return max_distance, 0

def compute_urban_neighbors(landuse_map,landuse_array, neighbor_matrix):
    # The objective is to maximize neighboring units between all landuses 1 "civil"
    # to 6 "living" and 3 "Industry"
    #identify all landuse_array civil (1)
    civil = np.where(landuse_array[:,1] == '1')
    civil_ids = landuse_array[civil,0].T

    living = np.where(landuse_array[:,1] == '6')
    living_ids = landuse_array[living, 0].T

    industry = np.where(landuse_array[:, 1] == '3')
    industry_ids = landuse_array[industry, 0].T

    neighbor_matrix_selection = neighbor_matrix[np.isin(neighbor_matrix[:,0], civil_ids)]
    #flatten all nested lists containing the neighbor ids
    selected_neighbors = neighbor_matrix_selection[:,1].T
    flattened_neighbors = []
    for i in range(selected_neighbors.shape[0]):
        flattened_neighbors = flattened_neighbors + selected_neighbors[i][0].tolist()
    #turn into np array
    flattened_neighbors = np.array(flattened_neighbors)

    #count the occurences
    count_neighbors_living = np.isin(flattened_neighbors, living_ids).sum()
    count_neighbors_industry = np.isin(flattened_neighbors, industry_ids).sum()
    return count_neighbors_living + count_neighbors_industry

def get_extreme_urban_neighbors(landuse_map, landuse_array, neighbor_matrix):
    landuse_array_c = copy.deepcopy(landuse_array)
    extreme_array = np.where(np.isin(landuse_array[:, 1], ["2", "4"]), "6", landuse_array[:, 1])
    landuse_array_c[:,1] = extreme_array
    max_urban_neighbors = compute_urban_neighbors(landuse_map,landuse_array_c, neighbor_matrix)
    return max_urban_neighbors, 0

def compute_agriculture_within_water_range(landuse_map,landuse_array, waters_buffer):
    # identify all landuse_array agriculture (4), intersect with water buffers and return area in ha
    return get_intersection_area(landuse_map, waters_buffer, filter = "landuse = '4' ")

def get_extreme_agriculture_within_water_range(landuse_map, landuse_array,waters_buffer):
    extreme_array = np.where(np.isin(landuse_array[:, 1], ["2", "3", "6"]), "4", landuse_array[:, 1])
    extreme_landuse_map = create_new_landuse_layer(landuse_map, extreme_array)
    max_agruculture_within_water_range = compute_agriculture_within_water_range(extreme_landuse_map,landuse_array, waters_buffer)
    return max_agruculture_within_water_range, 0

def compute_avg_agriculture_unit_size(landuse_map, landuse_array):
    avg_forest_area = get_avg_area_of_dissolved_filtered(landuse_map, "test.shp", multipoly=True, overwrite=False, filter = "landuse = '4' ")
    return avg_forest_area

def get_extreme_agri_unit_size(landuse_map, landuse_array):
    extreme_array = np.where(np.isin(landuse_array[:,1], ["2","3","6"]), "4", landuse_array[:,1])
    extreme_landuse_map = create_new_landuse_layer(landuse_map, extreme_array)
    max_area = compute_avg_agriculture_unit_size(extreme_landuse_map, landuse_array)
    return max_area, 0

