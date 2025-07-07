import logging
import os
from datetime import datetime
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from osgeo import gdal, ogr
from copy import deepcopy
from functools import reduce
from operator import concat
import pickle as pkl
from scipy.spatial import distance
import pandas as pd

def clone_data_to_mem(ds, name=''):
  if (isinstance(ds, gdal.Dataset)):
    return clone_raster_to_mem(ds, name)
  elif (isinstance(ds, ogr.DataSource)):
    return clone_vector_to_mem(ds, name)
  else:
    raise TypeError('Data source must be of GDAL dataset or OGR datasource')

def clone_vector_to_mem(vector_ds, name=''):
  driver = ogr.GetDriverByName('Memory')
  return driver.CopyDataSource(vector_ds, name,['OVERWRITE=YES'])

def clone_raster_to_mem(raster_ds, name=''):
  driver = gdal.GetDriverByName('MEM')
  return driver.CopyDataSource(raster_ds, name,['OVERWRITE=YES'])

def create_new_landuse_layer(vector_layer, land_use_array):
    #this populates the feature layer with the given land uses of the new solution in optimizaitons with vector data
    new_landuse_layer_ds = clone_data_to_mem(vector_layer, name=datetime.now().strftime("%d%H%M%S%f"))
    new_landuse_layer = new_landuse_layer_ds.GetLayer(0)
    # the new layer can be directly accessed via the handle pipes_mem or as source.GetLayer('pipes'):
    new_landuse_layer.CreateField(ogr.FieldDefn("landuse", ogr.OFTInteger))
    for i, item in enumerate(new_landuse_layer):
        item.SetField('landuse', int(np.nan_to_num(land_use_array[i][0])))
        new_landuse_layer.SetFeature(item)
    return new_landuse_layer_ds

def save_landuse_layer_geojson(landuse_map, path):
    layer = landuse_map.GetLayer(0)
    ref = layer.GetSpatialRef()
    #outShapefile = os.path.join( r"C:\Users\morit\PycharmProjects\GP\output_data\maps\out.shp")
    outShapefile = os.path.join(path, 'out.shp')
    outDriver = ogr.GetDriverByName("ESRI Shapefile")

    # Remove output shapefile if it already exists
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)

    # Create the output shapefile
    outDataSource = outDriver.CreateDataSource(outShapefile)
    outLayer = outDataSource.CreateLayer("out", ref, geom_type=ogr.wkbPolygon)

    idField = ogr.FieldDefn("oid", ogr.OFTString)
    outLayer.CreateField(idField)
    landuseField = ogr.FieldDefn("landuse", ogr.OFTString)
    outLayer.CreateField(landuseField)
    featureDefn = outLayer.GetLayerDefn()
    outfeature = ogr.Feature(featureDefn)

    for feature in layer:
        outfeature.SetGeometry(feature.GetGeometryRef())
        outfeature.SetField("oid", feature.GetField("oid"))
        id = feature.GetField("oid")
        lu = feature.GetField("landuse")
        outfeature.SetField("landuse", feature.GetField("landuse"))
        outLayer.CreateFeature(outfeature)
        feature = None
    outfeature = None
    outLayer, outDataSource, layer = None, None, None
    return outShapefile

def iteratively_fill_farthest_units(landuse_map, distance_matrix_points, landuse, min_percent, constrained_landuses):
    layer = landuse_map.GetLayer(0)
    total_area = 0
    area_per_unit = []
    for feature in layer:
        id = feature.GetField("oid")
        try:
            landuse = feature.GetField("landuse_re")
        except:
            landuse = feature.GetField("landuse")
        area = feature.geometry().Area()
        total_area += area
        area_per_unit.append([id,area, str(landuse)])

    area_per_unit_df = pd.DataFrame(area_per_unit, columns=["id", "area", "landuse"])
    grouped_areas = area_per_unit_df.groupby(['id', 'landuse']).mean()
    grouped_areas  = grouped_areas.reset_index()
    merged = distance_matrix_points.merge(grouped_areas, left_on='id', right_on='id')
    selected = merged[~merged["landuse"].isin(constrained_landuses.astype(str))]
    sorted = selected.sort_values(by=['distance'], ascending=False)
    sorted["summed_area"] = sorted.area.cumsum()
    sorted['index'] = sorted.sort_index().index
    sorted["less"] = sorted["summed_area"] < (total_area / min_percent)
    nr_less = sorted["less"].sum()
    ids_to_set = (sorted.iloc[0:nr_less].id).values

    distance_matrix_selection = distance_matrix_points[np.isin(distance_matrix_points['id'].values, ids_to_set)]
    # get average total distance for multipolygons
    grouped_distances = distance_matrix_selection.groupby(['id']).mean()
    # return mean distance in km
    mean_distance = grouped_distances['distance'].mean() / 1000
    return mean_distance

def get_intersection_area(ds1, ds2, filter):
    memory_driver = ogr.GetDriverByName('memory')
    memory_ds = memory_driver.CreateDataSource('temp')
    result_lyr = memory_ds.CreateLayer('result')
    ds1.GetLayer(0).SetAttributeFilter(filter)
    ds1.GetLayer(0).Intersection(ds2.GetLayer(0),result_lyr)
    area = 0
    for feat in result_lyr:
        area += feat.geometry().Area()
    ds1.GetLayer(0).SetAttributeFilter("")
    result_lyr.SetAttributeFilter("")
    memory_ds , result_lyr= None, None
    if area > 0:
        area = area/10000
    else:
        area = 0
    return area

def plot_vector_layer(ds, landuses):
    lyr = ds.GetLayer(0)
    # Get extent and calculate buffer size
    ext = lyr.GetExtent()
    xoff = (ext[1] - ext[0]) / 50
    yoff = (ext[3] - ext[2]) / 50

    # Prepare figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(ext[0] - xoff, ext[1] + xoff)
    ax.set_ylim(ext[2] - yoff, ext[3] + yoff)

    paths = []
    lyr.ResetReading()

    # Read all features in layer and store as paths
    for feat in lyr:
        geom = feat.geometry()
        codes = []
        all_x = []
        all_y = []
        for i in range(geom.GetGeometryCount()):
            # Read ring geometry and create path
            r = geom.GetGeometryRef(i)
            x = [r.GetX(j) for j in range(r.GetPointCount())]
            y = [r.GetY(j) for j in range(r.GetPointCount())]
            # skip boundary between individual rings
            codes += [mpath.Path.MOVETO] + \
                     (len(x) - 1) * [mpath.Path.LINETO]
            all_x += x
            all_y += y
        path = mpath.Path(np.column_stack((all_x, all_y)), codes)
        paths.append(path)

    c_ = []
    for i in range(landuses.shape[0]):
        if landuses[i][0] == 1:
            c_.append("bisque")
        elif landuses[i][0] == 2:
            c_.append("tan")
        elif landuses[i][0] == 3:
            c_.append("darkgoldenrod")
        if landuses[i][0] == 4:
            c_.append("gold")
        if landuses[i][0] == 5:
            c_.append("sienna")
        if landuses[i][0] == 6:
            c_.append("olivedrab")
        if landuses[i][0] == 7:
            c_.append("green")

    # Add paths as patches to axes
    for i in range(len(paths)):
        patch = mpatches.PathPatch(paths[i], facecolor=c_[i], edgecolor='black')
        ax.add_patch(patch)

    ax.set_aspect(1.0)
    plt.show()

def import_landuse_vector_features(file, mem_layer_name):
    DriverName = "ESRI Shapefile"  # e.g.: GeoJSON, ESRI Shapefile
    indriver = ogr.GetDriverByName(DriverName)
    srcdb =  indriver.Open(file, 0)
    return clone_data_to_mem(srcdb, name= mem_layer_name)

def get_polygon_centroids(landuse_layer):
    centroids = []
    for feature in landuse_layer:
        centroids.append(list(feature.geometry().Centroid().GetPoint()[0:2] ) )
    return centroids

def get_touching_neighbor_units(df):
    df = df.explode(index_parts=True)
    all_neighbors = []
    for index, row in df.iterrows():
        try:
            neighbors = np.array([row.oid, np.array(df[df.geometry.touches(row['geometry'])].oid.tolist())],dtype=object)
        #for invalid geometries buffer(0) might be needed
        except:
            neighbors = np.array(
                [row.oid, np.array(df[df.geometry.buffer(0).touches(row['geometry'].buffer(0))].oid.tolist())],
                dtype=object)
        all_neighbors.append(neighbors)
    all_neighbors = np.array(all_neighbors)

    all_neighbors = all_neighbors[all_neighbors[:, 0].argsort()]
    #return unique ids of multipolygons
    unique_units = np.split(all_neighbors[:, 1], np.unique(all_neighbors[:, 0], return_index=True)[1][1:])
    ids = np.unique(all_neighbors[:,0])
    #stack with unique ids and transpose
    all_neighbors = np.vstack((ids,unique_units)).T
    return all_neighbors

def get_neighbor_matrix(path, geodataframe):
    if os.path.exists(os.path.join(path,'neighbors.pkl')):
        with open(os.path.join(path,'neighbors.pkl'), 'rb') as inp:
            neighbors = pkl.load(inp)
    else:
        #set an id for each unique oid - landuse pair, adapt the rook 1 function accordingly
        geodataframe['gid'] = (geodataframe.groupby(['oid', 'landuse_re']).cumcount() == 0).astype(int)
        geodataframe['gid'] = geodataframe['gid'].cumsum()
        neighbors = get_touching_neighbor_units(geodataframe)
        with open(os.path.join(path,'neighbors.pkl'), 'wb') as outp:
            pkl.dump(neighbors, outp, pkl.HIGHEST_PROTOCOL)
    return neighbors

def get_distances_from_centroids_to_points(landuse_map, points, path):
    if os.path.exists(os.path.join(path,'point_distances.pkl')):
        with open(os.path.join(path,'point_distances.pkl'), 'rb') as inp:
            grouped_distances = pkl.load(inp)
    else:
        points = points.explode(index_parts=True)
        centroids = landuse_map.centroid
        coordinates_centroids = [[c.x, c.y] for c in centroids if c is not None]
        coordinates_points = [[p.x, p.y] for p in points.geometry]
        all_distances = []
        for c in range(len(coordinates_centroids)):
            total_distance_of_centroid_to_all_points = 0
            for p in coordinates_points:
                total_distance_of_centroid_to_all_points += distance.euclidean(coordinates_centroids[c], p)
            all_distances.append([landuse_map.iloc[c].oid, (total_distance_of_centroid_to_all_points/1000)])

        df = pd.DataFrame(all_distances, columns=['id', 'distance'])
        grouped_distances = df.groupby(['id']).mean()
        grouped_distances = grouped_distances.reset_index()
        with open(os.path.join(path,'point_distances.pkl'), 'wb') as outp:
            pkl.dump(grouped_distances, outp, pkl.HIGHEST_PROTOCOL)
    return grouped_distances

def get_minimum_average_distances_from_boundaries_to_waters(landuse_layer, waters_layer, path):
    if os.path.exists(os.path.join(path,'water_distances.pkl')):
        with open(os.path.join(path,'water_distances.pkl'), 'rb') as inp:
            grouped_distances = pkl.load(inp)
    else:
        ids = []
        closest_distances = []
        for lu_feature in landuse_layer:
            ids.append(lu_feature.GetField("oid"))
            distances = []
            for w_feature in waters_layer:
                lu_geom = lu_feature.GetGeometryRef()
                w_geom = w_feature.GetGeometryRef()
                distances.append(lu_geom.Distance(w_geom))
            closest_distances.append(min(distances))

        df = pd.DataFrame(np.array([ids, closest_distances]).T, columns=['id', 'distance'])
        df['distance'] = df['distance'].astype(float)
        grouped_distances = df.groupby(['id']).mean()
        grouped_distances = grouped_distances.reset_index()
        with open(os.path.join(path, 'water_distances.pkl'), 'wb') as outp:
            pkl.dump(grouped_distances, outp, pkl.HIGHEST_PROTOCOL)
    return grouped_distances

def createDS(ds_name, ds_format, geom_type, srs, overwrite=False):
    drv = ogr.GetDriverByName(ds_format)
    if os.path.exists(ds_name) and overwrite is True:
        ds_name = ogr.Open(ds_name)
        ds_name = None
    ds = drv.CreateDataSource(ds_name)
    lyr_name = os.path.splitext(os.path.basename(ds_name))[0]
    lyr = ds.CreateLayer(lyr_name, srs, geom_type)
    return ds, lyr

def get_landuse_array(landuse_map, landuse_column_name = "landuse"):
    landuse_layer = landuse_map.GetLayer(0)
    landuses = []
    ids = []
    for feature in landuse_layer:
        landuses.append(int(feature.GetField(landuse_column_name)))
        ids.append(feature.GetField("oid"))
    ids = np.array(ids)
    landuses = np.array(landuses, dtype = 'i')
    landuse_array = np.column_stack([ids, landuses])
    return landuse_array

def landuse_array_to_landusemap(landuse_map, landuse_array):
    # create an output datasource in memory
    outdriver = ogr.GetDriverByName('MEMORY')
    landuse_ds = outdriver.CreateDataSource('memData')

    # open the memory datasource with write access
    landuse_tmp = outdriver.Open('memData', 1)

    # copy a layer to memory
    landuse_mem = landuse_ds.CopyLayer(landuse_map.GetLayer(0),"landuse", ['OVERWRITE=YES'])

    # the new layer can be directly accessed via the handle pipes_mem or as source.GetLayer('pipes'):
    layer = landuse_ds.GetLayer(0)
    for feature in layer:
        oid = feature.GetField("oid")
        landuse = landuse_array[np.where(landuse_array[:, 0] == oid), 1][0][0]
        feature.SetField("landuse", landuse)

    landuse_map = None
    layer = None

    return landuse_tmp

def total_area_of_filtered_osgeo_layer(layer, filter):
    layer.SetAttributeFilter(filter)
    area = 0
    for feat in layer:
        if feat.geometry():
            area += feat.geometry().Area()
    layer = None
    return area

def createBuffer(inputds, outputBufferfn, bufferDist):
    inputlyr = inputds.GetLayer(0)

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if not os.path.exists(outputBufferfn):
        outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
        bufferlyr = outputBufferds.CreateLayer(outputBufferfn, geom_type=ogr.wkbPolygon)
        featureDefn = bufferlyr.GetLayerDefn()

        for feature in inputlyr:
            ingeom = feature.GetGeometryRef()
            geomBuffer = ingeom.Buffer(bufferDist)

            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(geomBuffer)
            bufferlyr.CreateFeature(outFeature)
            outFeature = None
    else:
        outputBufferds = ogr.Open(outputBufferfn)
    return outputBufferds

def get_feature_count(lyr):
    #return lyr.GetFeatureCount()
    # Vector input
    if hasattr(lyr, 'GetFeatureCount'):
        return lyr.GetFeatureCount()
    # Raster input (NumPy array)
    elif isinstance(lyr, np.ndarray):
        return lyr.size
    else:
        raise TypeError(f"Unsupported type in get_feature_count: {type(lyr)}")

def get_avg_area_of_dissolved_filtered(ds, output, multipoly=False, overwrite=False,filter = "landuse = '4' "):
    lyr = ds.GetLayer(0)
    out_ds, out_lyr = createDS(output, ds.GetDriver().GetName(), lyr.GetGeomType(), lyr.GetSpatialRef(), overwrite)
    defn = out_lyr.GetLayerDefn()
    multi = ogr.Geometry(ogr.wkbMultiPolygon)
    lyr.SetAttributeFilter(filter)
    for feat in lyr:
        if feat.geometry():
            feat.geometry().CloseRings() # this copies the first point to the end
            wkt = feat.geometry().ExportToWkt()
            multi.AddGeometryDirectly(ogr.CreateGeometryFromWkt(wkt))
    union = multi.UnionCascaded()

    out_feat = ogr.Feature(defn)
    out_feat.SetGeometry(union)
    out_lyr.CreateFeature(out_feat)

    area = 0
    for feat in out_lyr:
        area += feat.geometry().Area()

    area = area / 10000
    count = out_lyr.GetFeatureCount()
    if count > 0:
        avg_area_agriculture = area / count
    else:
        avg_area_agriculture  = 0

    out_ds.Destroy()
    out_ds = None
    out_lyr = None
    out_feat = None
    lyr = None
    return avg_area_agriculture

## post processing utility functions
def get_stats_of_optimization_runs(pf_dict):
    stats = []
    for key, value in pf_dict.items():
        if type(value) is dict:
            extreme_solutions = []
            runs = value["runs"]
            for k,v in runs.items():
                last_gen = list(v['generations'].items())[-1]
                pf_objective_values = []
                for item in last_gen[1]['pareto_front'].items:
                    pf_objective_values.append([i for i in item.fitness.wvalues])
                pf_objective_values = np.array(pf_objective_values)
                extreme_solutions.append([pf_objective_values[:, 0].max(),pf_objective_values[:, 1].max(),pf_objective_values[:, 2].max(),pf_objective_values[:, 3].max()])
            extreme_solutions = np.array(extreme_solutions)
            stats.append([[0, extreme_solutions[:, 0].mean(),extreme_solutions[:, 0].std()],
                          [1, extreme_solutions[:, 1].mean(), extreme_solutions[:, 1].std()],
                          [2, extreme_solutions[:, 2].mean(), extreme_solutions[:, 2].std()],
                          [3, extreme_solutions[:, 3].mean(), extreme_solutions[:, 3].std()]])
    return np.array(stats)

class Solution:
    _id = 0
    def __init__(self, representation, objective_values,landuse_order):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation
        self.objective_values = objective_values
        self.landuse_order = landuse_order

def get_optimal_solutions_pymoo(pf_dict, run = None):
    solutions = []

    pf = pf_dict["runs"][run]['pareto_front']
    for i in range(len(pf['fitness'])):
        solutions.append(
            Solution(representation=pf['representation'][i], objective_values=pf['fitness'][i]*-1, landuse_order= [1,2,3,4,5,6,7]))
    return solutions

def get_optimal_solutions(pf_dict, run = None):
    solutions = []

    for key, value in pf_dict.items():
        try:
            if type(value) is dict:
                if run is None:
                    runs = value["runs"]
                else:
                    runs = value["runs"][run]
                for k, v in runs.items():
                    last_gen = list(v['generations'].items())[-1]
                    for item in last_gen[1]['pareto_front'].items:
                        solutions.append(Solution(representation=item, objective_values=[i for i in item.fitness.values],
                                                  landuse_order=item.land_use_order))
        except:
            if run is None:
                runs = value
                for k, v in runs.items():
                    last_gen = list(v['generations'].items())[-1]
                    for item in last_gen[1]['pareto_front'].items:
                        solutions.append(
                            Solution(representation=item, objective_values=[i for i in item.fitness.values],
                                     landuse_order=item.land_use_order))

            else:
                runs = value[run]
                last_gen = list(runs['generations'].items())[-1]
                for item in last_gen[1]['pareto_front'].items:
                    solutions.append(
                        Solution(representation=item, objective_values=[i for i in item.fitness.values],
                                 landuse_order=item.land_use_order))
            return solutions

def add_input_data_uncertainty_per_cell(landuse_rasterarray, error_matrix, transition_matrix, seed):

    # get all unique landclasses (need to be integers)
    unique_landclasses = np.unique(landuse_rasterarray)
    # unique_patchids = np.unique(landusemap_patchids)
    #unique_landclasses = unique_landclasses[(unique_landclasses != landusemap_emptycellvalue)]
    landuse_rasterarray = landuse_rasterarray.copy()

    # iterate through input land use raster. Reassign the cell value corresponding to the error matrix, which includes all probabilities
    # that a class was misclassified as another class.
    for row_index in range(landuse_rasterarray.shape[0]):
        for col_index in range(landuse_rasterarray.shape[1]):
            row_index_error_matrix = np.where(error_matrix[:, 0] == landuse_rasterarray[row_index, col_index])[0]

            # filter non_existant land use map values
            mask_required_uncertainty_values = np.isin(error_matrix[0, :], unique_landclasses)
            relevant_error_matrix_row = error_matrix[row_index_error_matrix[0], :]
            relevant_error_matrix_row = relevant_error_matrix_row[mask_required_uncertainty_values]

            # reassign the value to probabilistic value from error matrix
            try:
                reassigned_value = int(seed.choice(unique_landclasses, 1, p=relevant_error_matrix_row))
                # make sure that only constrained cells are reclassified due to min-max-regulations
                trans_matrix_col_oldvalue = \
                (np.where(transition_matrix[:, 0] == landuse_rasterarray[row_index, col_index])[0])[0]
                trans_matrix_col_newvalue = (np.where(transition_matrix[:, 0] == reassigned_value)[0])[0]
                trans_matrix_row_oldvalue = \
                (np.where(transition_matrix[0, :] == landuse_rasterarray[row_index, col_index])[0])[0]
                trans_matrix_row_newvalue = (np.where(transition_matrix[0, :] == reassigned_value)[0])[0]
                if transition_matrix[trans_matrix_row_oldvalue, trans_matrix_col_newvalue] == 0 or transition_matrix[
                    trans_matrix_row_newvalue, trans_matrix_col_oldvalue] == 0:
                    landuse_rasterarray[row_index, col_index] = reassigned_value
            except:
                print("reassigning did not work. check if all possible landuse map cell values are in use")
    return landuse_rasterarray
