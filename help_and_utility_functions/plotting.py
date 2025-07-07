import numpy as np
from deap import gp
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from optimization_setup.map_translation_and_validation import validate_landuse_map, convert_function_to_map
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from help_and_utility_functions.utility_functions import save_landuse_layer_geojson,create_new_landuse_layer

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib import cm
import seaborn as sns
from pathlib import Path
import pandas as pd
import os

import geopandas as gpd

# Prettier plotting with seaborn
sns.set(font_scale=1.5, style="white")

def get_solution_tree_axis(individual, ax):
    #ax.set_title("Tree representation")
    # plot tree
    nodes, edges, labels = gp.graph(individual.representation)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")
    # draw all nodes homogeneously, and edge weights as filtered
    nx.draw_networkx_nodes(g, pos, ax=ax)
    nx.draw_networkx_edges(g, pos, ax=ax)
    nx.draw_networkx_labels(g, pos, labels, ax=ax)

def plot_raster_solution(landuse_problem, individual, title):
    def plot_maps(solution_map,ax, colormaps, length):
        """
        Helper function to plot data with associated colormap.
        """
        #n = len(colormaps)
        for cmap in colormaps:
            psm = ax.pcolormesh(solution_map, cmap=cmap, rasterized=True, vmin=np.min(solution_map), vmax=np.max(solution_map))
            #fig.colorbar(psm, ax=ax)
        return psm

    def get_solution_map_ax(solution_landuse_map, ax, title):
        viridis = cm.get_cmap('viridis', 256)
        cm_continuous_map = viridis(np.linspace(0, 1, 256))
        cropland_1 = np.array([256 / 256, 246 / 256, 143 / 256, 1])
        cropland_2 = np.array([238 / 256, 230 / 256, 133 / 256, 1])
        cropland_3 = np.array([256 / 256, 215 / 256, 1 / 256, 1])
        cropland_4 = np.array([238 / 256, 201 / 256, 0 / 256, 1])
        cropland_5 = np.array([205 / 256, 173 / 256, 1 / 256, 1])
        pasture = np.array([162 / 256, 205 / 256, 90 / 256, 1])
        forest = np.array([34 / 256, 139 / 256, 34 / 256, 1])
        urban = np.array([105 / 256, 105 / 256, 105 / 256, 1])
        clist = [cropland_1, cropland_2,cropland_3,cropland_4,cropland_5,forest,pasture,urban]
        cmp_solution_map = ListedColormap(clist)
        actual_solution_plot = plot_maps(solution_landuse_map,ax=ax, colormaps=[cmp_solution_map], length = len(clist))
        if title is not None:
            ax.set_title(title)

        legend_labels = {"#fff68fff": "Cropland 1",
                         "#eee685ff": "Cropland 2",
                         "#ffd700ff": "Cropland 3",
                         "#eec800ff": "Cropland 4",
                         "#ccaa01ff": "Cropland 5",
                         "#228b22ff": "Forest",
                         "#a2cd5aff": "Pasture",
                         "#696969ff": "Urban"
                         }

        patches = [Patch(color=color, label=label)
                   for color, label in legend_labels.items()]

        ax.legend(handles=patches, facecolor="white", loc='center left', bbox_to_anchor=(1, 0.5))
        return actual_solution_plot

    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(24, 12),gridspec_kw={'width_ratios': [1, 1]})

    func = landuse_problem.toolbox.compile(expr=individual.representation)

    # convert function into landuse map. If vector optimization a landuse_array(numpy) is also returned, otherwise it is None
    landuse_map = convert_function_to_map(func, landuse_problem.mapping_points, landuse_problem, individual.landuse_order)

    # validate landuse map. If valid, go to evaluation
    validity, landuse_map, landuse_array = validate_landuse_map(landuse_map, landuse_problem.constraints)

    get_solution_tree_axis(individual, ax1)
    get_solution_map_ax(landuse_map, ax2, title = None)
    ax1.set_axis_off()
    ax2.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.title(landuse_problem.name)
    fig.suptitle(title, fontsize=16)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    print()


def plot_pf(obj1, obj2, title = None, xlabel= None, ylabel=None):
    plt.scatter(obj1, obj2, alpha=0.5)
    plt.title(title)
    plt.xlabel=xlabel
    plt.ylabel = ylabel
    plt.show()

def plot_vector_solution(landuseproblem, individual, title):
    def get_solution_map_ax(solution_landuse_map, ax):
        parentdir = Path(__file__).parent.parent
        path_map = save_landuse_layer_geojson(solution_landuse_map, path=os.path.join(parentdir, 'output_data', 'maps'))
        plot_vector_landuse_map_geopandas(path_map,ax, cathegorical=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [1, 2]})


    try:
        func = landuseproblem.toolbox.compile(expr=individual.representation)
        # in case of gp it goes here
        # convert function into landuse map. If vector optimization a landuse_array(numpy) is also returned, otherwise it is None
        landuse_map = convert_function_to_map(func, landuseproblem.mapping_points, landuseproblem, individual.landuse_order)
    except:
        landuse_map = create_new_landuse_layer(landuseproblem.initial_landuse_map_datasource, individual.representation.astype(str))


    # validate landuse map. If valid, go to evaluation
    validity, landuse_map, landuse_array = validate_landuse_map(landuse_map, landuseproblem.constraints)
    get_solution_tree_axis(individual, ax1)
    get_solution_map_ax(landuse_map, ax2)

    ax1.set_axis_off()
    ax2.set_axis_off()

    waters = landuseproblem.additional_data["geodataframe_waters"].plot( color = 'blue', ax= ax2, legend=True, edgecolor="darkblue")
    wka = landuseproblem.additional_data["geodataframe_wka"].plot(color='yellow', ax=ax2, legend=True, markersize=5)

    #map axis title
    plt.title("Solution vector map")
    fig.suptitle(title, fontsize=16)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def plot_vector_landuse_map_geopandas(path, ax, cathegorical = True):
    #to plot solutions call for example:
    # from utility_functions import save_landuse_layer_geojson
    # path_map = save_landuse_layer_geojson(landuse_map, path = r"input_data\vector_optimization\100_fluren")
    # plot_solutions.plot_vector_landuse_map_geopandas(path_map)
    map = gpd.read_file(path)
    map["landuse_name"] = map["landuse"].map({'1': 'Civil', '2': 'Rural non-forest',
                                              '3': 'Industry', '4': 'Agriculture', '5': 'Forest',
                                              '6': 'Living', '7': 'Transport and water'})
    if cathegorical:
        symb = {'Civil': 'orange',
                       'Rural non-forest': 'lightgreen',
                       'Industry': 'burlywood',
                       'Agriculture': 'brown',
                       'Forest': 'darkgreen',
                       'Living': 'red',
                       'Transport and water': 'grey'}
        map.plot(color=map["landuse_name"].map(symb), linewidth=.1, edgecolor='0.2',
             legend=True, legend_kwds={'bbox_to_anchor':(1.25, 1.05),'fontsize':16,'frameon':False}, ax = ax)

        custom_points = [Line2D([0], [0], marker="o", linestyle="none", markersize=5, color=color) for color in
                         symb.values()]
        leg_points = ax.legend(custom_points, symb.keys(), title="Landuses", loc=(1.1, .1))
        ax.add_artist(leg_points)
    else:
        map.plot()

def plot_single_objective_extremes(landuse_problem, optimal_solutions, optimal_solutions_objective_values, objectives = None):
    if objectives is None:
        obj = landuse_problem.objectives
    else:
        obj = [landuse_problem.objectives[o] for o in objectives]
    for ido, o in enumerate(obj):
        if o.minimization is False:
            best = np.where(optimal_solutions_objective_values[:, ido] == optimal_solutions_objective_values[:, ido].max())[0][0]
        else:
            best = np.where(optimal_solutions_objective_values[:, ido] == optimal_solutions_objective_values[:, ido].min())[0][0]
        landuse_problem.plot_solution_map(landuse_problem,
                                              individual=optimal_solutions[best], title = "Best solution of objective {}".format(o.name))

def plot_2d_pareto_fronts(landuse_problem, non_dominated_solutions):
    def non_dominated(obj_values, maximise=True):
        is_efficient = np.ones(obj_values.shape[0], dtype=bool)
        for i, c in enumerate(obj_values):
            if is_efficient[i]:
                if maximise:
                    is_efficient[is_efficient] = np.any(obj_values[is_efficient] >= c, axis=1)  # Remove dominated points
                else:
                    is_efficient[is_efficient] = np.any(obj_values[is_efficient] <= c, axis=1)  # Remove dominated points
        return is_efficient

    obj = landuse_problem.objectives

    fig = make_subplots(rows=len(obj), cols=len(obj), column_titles= [o.name for o in obj],
                        row_titles=[o.name for o in obj],
                        start_cell='top-left', horizontal_spacing = 0.03, vertical_spacing=0.03)
    for id_i, i in enumerate(obj):
        for id_j, j in enumerate(obj):
            if id_i == id_j:
                fig.add_trace(
                    go.Scatter(y=non_dominated_solutions[:, id_i], x=non_dominated_solutions[:, id_i], mode='markers',
                               opacity=0.5,
                               marker=dict(color='LightSkyBlue')), row=id_i + 1, col=id_j + 1)

            if id_i < id_j:
                if i.minimization:
                    obj_values_i = non_dominated_solutions[:, id_i] * -1

                else:
                    obj_values_i = non_dominated_solutions[:, id_i]

                if j.minimization:
                    obj_values_j = non_dominated_solutions[:, id_j] * -1

                else:
                    obj_values_j = non_dominated_solutions[:, id_j]

                stacked = np.vstack((obj_values_i, obj_values_j)).T

                non_dominated_ids = non_dominated(stacked)



                fig.add_trace(
                    go.Scatter(y=non_dominated_solutions[:, id_i], x=non_dominated_solutions[:, id_j],
                               mode='markers', opacity=0.5,
                               marker=dict(color='LightSkyBlue')),
                    row=id_i + 1, col=id_j + 1)

                fig.add_trace(go.Scatter(y=non_dominated_solutions[non_dominated_ids][:, id_i],
                                         x=non_dominated_solutions[non_dominated_ids][:, id_j],
                                         mode='markers', opacity=1,
                                         marker=dict(color='darkblue')),
                              row=id_i + 1, col=id_j + 1)

                try:
                    fig.add_hline(y=i.extremes["extreme_best"], line_dash="dash", row=id_i + 1, col=id_j + 1)
                    fig.add_vline(x=j.extremes["extreme_best"], line_dash="dash", row=id_i + 1, col=id_j + 1)
                except:
                    pass

    fig.update_layout(
        #title='Objective pair pareto fronts',
        width=1400,
        height=1400,
    )
    fig.update_xaxes(automargin=True)

    fig.show()

def non_dominated(obj_values, maximise=True):
    is_efficient = np.ones(obj_values.shape[0], dtype=bool)
    for i, c in enumerate(obj_values):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(obj_values[is_efficient] > c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(obj_values[is_efficient] < c, axis=1)  # Remove dominated points
    return is_efficient

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs>costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def plot_2d_uncertain_pareto_fronts(landuse_problem, non_dominated_solutions_arrays):


    obj = landuse_problem.objectives

    fig = make_subplots(rows=len(obj), cols=len(obj), column_titles= [o.name for o in obj],
                        row_titles=[o.name for o in obj],
                        start_cell='top-left', horizontal_spacing = 0.03, vertical_spacing=0.03)
    for id_i, i in enumerate(obj):
        for id_j, j in enumerate(obj):
            for k in range(len(non_dominated_solutions_arrays)):
                non_dominated_solutions = non_dominated_solutions_arrays[k]
                if id_i == id_j:
                    pass
                    # fig.add_trace(
                    #     go.Scatter(y=non_dominated_solutions[:, id_i], x=non_dominated_solutions[:, id_i], mode='markers',
                    #                opacity=0.5,
                    #                marker=dict(color='LightSkyBlue')), row=id_i + 1, col=id_j + 1)

                if id_i < id_j:
                    if i.minimization:
                        obj_values_i = non_dominated_solutions[:, id_i] * -1

                    else:
                        obj_values_i = non_dominated_solutions[:, id_i]

                    if j.minimization:
                        obj_values_j = non_dominated_solutions[:, id_j] * -1

                    else:
                        obj_values_j = non_dominated_solutions[:, id_j]

                    stacked = np.vstack((obj_values_i, obj_values_j)).T

                    non_dominated_ids = is_pareto_efficient(stacked)



                    # fig.add_trace(
                    #     go.Scatter(y=non_dominated_solutions[:, id_i], x=non_dominated_solutions[:, id_j],
                    #                mode='markers', opacity=0.5,
                    #                marker=dict(color='LightSkyBlue')),
                    #     row=id_i + 1, col=id_j + 1)

                    fig.add_trace(go.Scatter(y=non_dominated_solutions[non_dominated_ids][:, id_i],
                                             x=non_dominated_solutions[non_dominated_ids][:, id_j],
                                             mode='markers', opacity=1
                                             ),
                                  row=id_i + 1, col=id_j + 1)



    fig.update_layout(
        #title='Objective pair pareto fronts',
        width=1400,
        height=1400,
    )
    fig.update_xaxes(automargin=True)

    fig.show()

def plot_2d_uncertain_pareto_fronts2(landuse_problem, non_dominated_solutions_arrays):

    obj = landuse_problem.objectives

    fig = make_subplots(rows=len(obj), cols=len(obj), column_titles= [o.name for o in obj],
                        row_titles=[o.name for o in obj],
                        start_cell='top-left', horizontal_spacing = 0.03, vertical_spacing=0.03)
    all_fronts = []
    for id_i, i in enumerate(obj):
        for id_j, j in enumerate(obj):
            all_obj1 = []
            all_obj2 = []
            for k in range(len(non_dominated_solutions_arrays)):
                non_dominated_solutions = non_dominated_solutions_arrays[k]
                if id_i == id_j:
                    pass
                    # fig.add_trace(
                    #     go.Scatter(y=non_dominated_solutions[:, id_i], x=non_dominated_solutions[:, id_i], mode='markers',
                    #                opacity=0.5,
                    #                marker=dict(color='LightSkyBlue')), row=id_i + 1, col=id_j + 1)

                if id_i < id_j:
                    if i.minimization:
                        obj_values_i = non_dominated_solutions[:, id_i] * -1

                    else:
                        obj_values_i = non_dominated_solutions[:, id_i]

                    if j.minimization:
                        obj_values_j = non_dominated_solutions[:, id_j] * -1

                    else:
                        obj_values_j = non_dominated_solutions[:, id_j]

                    stacked = np.vstack((obj_values_i, obj_values_j)).T

                    non_dominated_ids = is_pareto_efficient(stacked)

                    all_obj1.append(non_dominated_solutions[non_dominated_ids][:, id_j])
                    all_obj2.append(non_dominated_solutions[non_dominated_ids][:, id_i])


            if len(all_obj1)>0:

                x = np.concatenate(all_obj1)
                y = np.concatenate(all_obj2)



                # fig.add_trace(go.Scatter(
                #     x=x,
                #     y=y,
                #     mode='markers',
                #     showlegend=False,
                #     marker=dict(
                #         opacity=0.4,
                #         color='white',
                #         size=6,
                #         line=dict(width=1),
                #     )
                # ),row=id_i + 1, col=id_j + 1)

                fig.add_trace(go.Histogram2d(
                    histnorm="probability",
                    x=x,
                    y=y,
                    colorscale='YlGnBu',
                    nbinsx=20,
                    nbinsy=20,
                    zauto=False,
                ),row=id_i + 1, col=id_j + 1)

    fig.update_layout(
        xaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20 ),
        yaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20 ),
        autosize=False,
        width=1400,
        height=1400,
        hovermode='closest',
    )

    fig.update_xaxes(automargin=True)

    fig.show()
    print()


def plot_2d_pareto_fronts_benchmark(landuse_problem, non_dominated_solutions, second_front):
    def non_dominated(obj_values, maximise=True):
        is_efficient = np.ones(obj_values.shape[0], dtype=bool)
        for i, c in enumerate(obj_values):
            if is_efficient[i]:
                if maximise:
                    is_efficient[is_efficient] = np.any(obj_values[is_efficient] >= c, axis=1)  # Remove dominated points
                else:
                    is_efficient[is_efficient] = np.any(obj_values[is_efficient] <= c, axis=1)  # Remove dominated points
        return is_efficient

    obj = landuse_problem.objectives

    fig = make_subplots(rows=len(obj), cols=len(obj), column_titles= [o.name for o in obj],
                        row_titles=[o.name for o in obj],
                        start_cell='top-left', horizontal_spacing = 0.03, vertical_spacing=0.03)
    for id_i, i in enumerate(obj):
        for id_j, j in enumerate(obj):
            if id_i == id_j:
                fig.add_trace(go.Scatter(y=non_dominated_solutions[:,id_i], x=non_dominated_solutions[:,id_i],mode='markers', opacity=0.5,
                               marker=dict(color='LightSkyBlue')), row=id_i + 1, col=id_j + 1)
                fig.add_trace(
                    go.Scatter(y=second_front[:, id_i], x=second_front[:, id_i], mode='markers',
                               opacity=0.5,
                               marker=dict(color='Red')), row=id_i + 1, col=id_j + 1)

            if id_i < id_j:
                if i.minimization:
                    obj_values_i = non_dominated_solutions[:,id_i] * -1
                    obj_values_i2 = second_front[:, id_i] * -1
                else:
                    obj_values_i = non_dominated_solutions[:,id_i]
                    obj_values_i2 = second_front[:, id_i]
                if j.minimization:
                    obj_values_j = non_dominated_solutions[:,id_j] * -1
                    obj_values_j2 = second_front[:, id_j] * -1
                else:
                    obj_values_j = non_dominated_solutions[:,id_j]
                    obj_values_j2 = second_front[:, id_j]
                stacked = np.vstack((obj_values_i, obj_values_j)).T
                stacked2 = np.vstack((obj_values_i2, obj_values_j2)).T
                non_dominated_ids = non_dominated(stacked)
                non_dominated_ids2 = non_dominated(stacked2)


                fig.add_trace(
                    go.Scatter(y=second_front[:, id_i], x=second_front[:, id_j],
                               mode='markers', opacity=0.5,
                               marker=dict(color='Red')),
                    row=id_i + 1, col=id_j + 1)

                fig.add_trace(
                    go.Scatter(y=non_dominated_solutions[:,id_i], x=non_dominated_solutions[:,id_j],
                               mode='markers', opacity=0.5,
                               marker=dict(color='LightSkyBlue')),
                    row=id_i + 1, col=id_j + 1)

                fig.add_trace(go.Scatter(y=second_front[non_dominated_ids2][:, id_i],
                                         x=second_front[non_dominated_ids2][:, id_j],
                                         mode='markers', opacity=1,
                                         marker=dict(color='Red')),
                              row=id_i + 1, col=id_j + 1)
                fig.add_trace(go.Scatter(y=non_dominated_solutions[non_dominated_ids][:,id_i], x=non_dominated_solutions[non_dominated_ids][:,id_j],
                                         mode='markers', opacity=1,
                                         marker=dict(color='darkblue')),
                                         row=id_i + 1, col=id_j + 1)
                #try:
                # fig.add_trace(go.Scatter(y=[i.extremes["extreme_best"]], x=[0], mode='markers',
                #                              marker=dict(color='black',symbol='x')), row=id_i + 1, col=id_j + 1)
                fig.add_hline(y=i.extremes["extreme_best"],line_dash="dash", row=id_i + 1, col=id_j + 1)
                fig.add_vline(x=j.extremes["extreme_best"],line_dash="dash", row=id_i + 1, col=id_j + 1)

                #except:
                #    pass

    fig.update_layout(
        #title='Objective pair pareto fronts',
        width=1400,
        height=1400,
    )
    fig.update_xaxes(automargin=True)

    fig.show()



def plot_run_times(fname):
    df = pd.read_csv(fname, delimiter=';')
    fig = px.scatter(df, x="Nr Cells", y=["Patches","Single cells"], trendline="ols",
                     labels={
                         "variable": "Land use representation",
                         "value": "Run time for 3 runs (minutes)"
                     }, )
    fig.show()

def plot_run_time_comparison_GP_CoMOLA(fname):
    df = pd.read_csv(fname, delimiter=';')
    fig = px.scatter(df, x="Problem size", y=["GP (in seconds)","Comola without repair mutation (in minutes)", "Comola with repair mutation (in minutes)"],
                     labels={
                         "variable": "Algorithm",
                         "value": "Run time"
                     }
                        , width=400, height=300
                     )
    fig.update_layout(showlegend=False)
    fig.show()

def plot_run_time_comparison_GP_NSGA3(fname):
    df = pd.read_csv(fname, delimiter=';')
    fig = px.scatter(df, x="Problem size", y=["GP (in minutes)","NSGA3 (in minutes)"],
                     labels={
                         "variable": "Algorithm",
                         "value": "Run time"
                     }
                     , width=400, height=300
                     )
    fig.update_layout(showlegend=False)
    fig.show()

def plot_unique_landuse_order(dict, runs = 0):
    gens = dict["runs"][runs]["generations"]
    count_unique = []
    for i in range(len(gens)):
        count_unique.append([i,np.unique(np.array([ind.land_use_order for ind in gens[i]["pareto_front"]]), axis = 0).shape[0]/ len(gens[i]["pareto_front"])*100])
    df = pd.DataFrame(count_unique, columns = ["gen", "percentage_unique"])
    fig = px.scatter(df, x="gen", y="percentage_unique",        labels={
                         "gen": "Generation",
                         "percentage_unique": "Percentage of unique land use orders"
                     },
                     )
    fig.show()