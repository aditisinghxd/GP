import numpy as np

def getNbh(col, row, ncols, nrows, four_neighbours):
    """Determine the neighboring cells of the cell (col,row) and
       return the coordinates as arrays separated in nbhs_col and nbhs_row.
       The combination of the elements gives the coordinates of the neighbouring cells.

       input:
           col and row are coordinates of the reviewed element
           ncols, nrows are numbers of rows and columns in the map
           four_neighbours if True than 4 neighboring cells are scanned else 8
    """

    # assuming that a cell in the center has 8 neighbouring cells
    if four_neighbours == 'False':
        # cell is no edge cell
        if col > 0 and row > 0 and row < nrows - 1 and col < ncols - 1:
            nbhs_col = [x + col for x in [-1, -1, -1, 0, 0, 1, 1, 1]]
            nbhs_row = [x + row for x in [-1, 0, 1, -1, 1, -1, 0, 1]]
        # cell is a left edge element but no corner element
        elif col == 0 and row > 0 and row < nrows - 1:
            nbhs_col = [x + col for x in [0, 1, 1, 0, 1]]
            nbhs_row = [x + row for x in [-1, -1, 0, 1, 1]]
            # cell is a right edge element but no corner element
        elif col == ncols - 1 and row > 0 and row < nrows - 1:
            nbhs_col = [x + col for x in [-1, -1, -1, 0, 0]]
            nbhs_row = [x + row for x in [-1, 0, 1, -1, 1]]
        # cell is an upper edge element but no corner element
        elif row == 0 and col > 0 and col < ncols - 1:
            nbhs_col = [x + col for x in [-1, -1, 0, 1, 1]]
            nbhs_row = [x + row for x in [0, 1, 1, 0, 1]]
        # cell is a bottom edge element but no corner element
        elif row == nrows - 1 and col > 0 and col < ncols - 1:
            nbhs_col = [x + col for x in [-1, -1, 0, 1, 1]]
            nbhs_row = [x + row for x in [-1, 0, -1, -1, 0]]
            # cell is in the left upper corner
        elif col == 0 and row == 0:
            nbhs_col = [x + col for x in [0, 1, 1]]
            nbhs_row = [x + row for x in [1, 0, 1]]
        # cell is in the left bottom corner
        elif col == 0 and row == nrows - 1:
            nbhs_col = [x + col for x in [0, 1, 1]]
            nbhs_row = [x + row for x in [-1, 0, -1]]
            # cell is in the right upper corner
        elif col == ncols - 1 and row == 0:
            nbhs_col = [x + col for x in [-1, -1, 0]]
            nbhs_row = [x + row for x in [0, 1, 1]]
        # cell is in the right bottom corner
        else:
            nbhs_col = [x + col for x in [-1, -1, 0]]
            nbhs_row = [x + row for x in [-1, 0, -1]]

            # assuming that a cell in the center has 4 neighbouring cells
    elif four_neighbours == 'True':
        # cell is no edge cell
        if col > 0 and row > 0 and row < nrows - 1 and col < ncols - 1:
            nbhs_col = [x + col for x in [-1, 0, 0, 1]]
            nbhs_row = [x + row for x in [0, -1, 1, 0]]
        # cell is a left edge element but no corner element
        elif col == 0 and row > 0 and row < nrows - 1:
            nbhs_col = [x + col for x in [0, 1, 0]]
            nbhs_row = [x + row for x in [-1, 0, 1]]
            # cell is a right edge element but no corner element
        elif col == ncols - 1 and row > 0 and row < nrows - 1:
            nbhs_col = [x + col for x in [-1, 0, 0]]
            nbhs_row = [x + row for x in [0, 1, -1]]
            # cell is an upper edge element but no corner element
        elif row == 0 and col > 0 and col < ncols - 1:
            nbhs_col = [x + col for x in [-1, 0, 1]]
            nbhs_row = [x + row for x in [0, 1, 0]]
        # cell is an bottom edge element but no corner element
        elif row == nrows - 1 and col > 0 and col < ncols - 1:
            nbhs_col = [x + col for x in [-1, 0, 1]]
            nbhs_row = [x + row for x in [0, -1, 0]]
            # cell is in the left upper corner
        elif col == 0 and row == 0:
            nbhs_col = [x + col for x in [0, 1]]
            nbhs_row = [x + row for x in [1, 0]]
        # cell is in the left bottom corner
        elif col == 0 and row == nrows - 1:
            nbhs_col = [x + col for x in [0, 1]]
            nbhs_row = [x + row for x in [-1, 0]]
            # cell is in the right upper corner
        elif col == ncols - 1 and row == 0:
            nbhs_col = [x + col for x in [-1, 0]]
            nbhs_row = [x + row for x in [0, 1]]
        # cell is in the right bottom corner
        else:
            nbhs_col = [x + col for x in [-1, 0]]
            nbhs_row = [x + row for x in [0, -1]]

    else:
        msg = "Error: ini input for four_neighbours is not correct. Please check."
        print(msg)
        raise SystemError("Error: ini input for four_neighbours is not correct")

    return [nbhs_row, nbhs_col]

def determine_static_classes(trans_matrix, max_range):
    """This function determines all classes which are excluded from optimization (static elements)
       and returns arrays with the indices of static and non static elements.

       input:
           trans_matrix holding the land use transition constraints
           max_range is the maximum number of possible land use options
    """
    # identify all classes where column and row elements are zero (apart from diagonal elements)
    # that means that all patches of these classes cannot be converted
    static_elements = []
    nonstatic_elements = []
    # filter columns which fulfill the condition
    ones = 0
    # row-wise check for ones
    for row in range(1, trans_matrix.shape[0]):
        for col in range(1, trans_matrix.shape[1]):
            if trans_matrix[row][col] == 1:
                ones += 1
        # mark the candidate as static or non static element (row is checked)
        # if ones for row = 1 or max_range < land use index of trans_matrix
        if ones == 1 or trans_matrix[row][0] > max_range:
            static_elements.append(trans_matrix[row][0])
        else:
            nonstatic_elements.append(trans_matrix[row][0])
        ones = 0

    # column-wise check for ones
    ones = 0
    index = 0
    if len(static_elements) != 0:
        for col in range(1, trans_matrix.shape[1]):
            if index < len(static_elements) and static_elements[index] <= max_range:
                if trans_matrix[0][col] == static_elements[index]:
                    for row in range(1, trans_matrix.shape[0]):
                        if trans_matrix[row][col] == 1:
                            ones += 1
                    if ones != 1:
                        # index remains as it is for the next round
                        # because of length reduction from static_element
                        nonstatic_elements.append(static_elements[index])
                        del static_elements[index]
                    else:
                        index += 1
                    ones = 0
                if len(static_elements) == 0:
                    break

    return static_elements, nonstatic_elements


def determine_IDmap_patch_elements(row, col, patch_map, map, neighbors, cls, landuse, error_dic, four_neighbours):
    """This recursive function scans all patch elements of the patch ID map,
       check if all patch elements have the same land use index
       and return the coordinates of the patch elements.

       input:
           col and row are coordinates of the parent element
           patch_map is the given patch ID map
           map is the original ascii map
           neighbors is a matrix for marking the scanned cells
           cls is the patch index
           landuse is the index of the first scanned patch element
           four_neighbours if True than 4 neighboring cells are scanned else 8
    """
    # determine coordinates of neighboring cells
    new_nbhs_row, new_nbhs_col = getNbh(col, row, map.shape[1], map.shape[0], four_neighbours)
    # stack for patch elements whose neighboring cells should be determined
    nbhs_row = []
    nbhs_col = []
    for i in range(len(new_nbhs_row)):
        # add new neighboring cells to nbhs_row/col if new cells belong to cls in patch_map and are not jet marked as scanned
        if patch_map[new_nbhs_row[i], new_nbhs_col[i]] == cls and neighbors[new_nbhs_row[i], new_nbhs_col[i]] == 0:
            nbhs_row.append(new_nbhs_row[i])
            nbhs_col.append(new_nbhs_col[i])
    # print ("nbhs_row, nbhs_col von (%s,%s): %s, %s" %(row, col, nbhs_row, nbhs_col))
    while len(nbhs_row) > 0:
        # if cell was not scanned before
        if neighbors[nbhs_row[0], nbhs_col[0]] == 0:
            # mark this cell in neighbors with True (scanned)
            neighbors[nbhs_row[0], nbhs_col[0]] = 1
            # and check if land use type for all patch elements is equal
            if map[nbhs_row[0], nbhs_col[0]] != landuse:
                error_dic.update({"(%s,%s)" % (nbhs_row[0], nbhs_col[0]): "more than one land use index for one patch"})
            # determine coordinates of neighboring cells from this cell
            new_nbhs_row, new_nbhs_col = getNbh(nbhs_col[0], nbhs_row[0], map.shape[1], map.shape[0], four_neighbours)
            for i in range(len(new_nbhs_row)):
                # add new neighboring cells to nbhs_row/col if new cells belong to cls in patch_map and not jet marked as scanned
                if patch_map[new_nbhs_row[i], new_nbhs_col[i]] == cls and neighbors[
                    new_nbhs_row[i], new_nbhs_col[i]] == 0:
                    nbhs_row.append(new_nbhs_row[i])
                    nbhs_col.append(new_nbhs_col[i])
                    # delete this checked neighboring cell of the array
        del nbhs_row[0]
        del nbhs_col[0]

    return neighbors, error_dic

def read_patch_ID_map(patches, map, NODATA_value, static_elements, four_neighbours):
    """This function reads a given patch ID map, checks its plausibility
       and returns the patch ID map as a 2 dimensional array and the start individual as vector.

       input:
           file with the patch ID map (ascii format)
           map is the original ascii map
           NODATA_value is the NODATA_value of the original map
           static_elements are the land use indices excluded from the optimization
           four_neighbours if True than 4 neighboring cells are scanned else 8
    """
    # transform map into a matrix

    # check if number of rows and columns are equal to the original map
    if (map.shape[0] != patches.shape[0]) or (map.shape[1] != patches.shape[1]):
        msg = "Error: Number of rows or columns of the original map (rows: %s, columns: %s) and the patch ID map (rows: %s, columns: %s) are not equal. Please check." % (
        map.shape[0], map.shape[1], patches.shape[0], patches.shape[1])
        print(msg)
    # check that land use indices of the patch elements are equal
    # and that the land use index is not a static element
    # if all checks are okay then add the land use index to the genome of the start individual
    max_patchID = patches.max()
    genom = np.zeros(max_patchID, int)
    help_map = np.zeros([map.shape[0], map.shape[1]], bool)  # default is 0/False
    error_dic = {}
    for row in range(0, help_map.shape[0]):
        for col in range(0, help_map.shape[1]):
            # map element was not scanned before
            if help_map[row, col] == 0:
                # first case: patch element = 0
                # check if the index in the original map is a static element or NODATA_value
                if patches[row, col] == 0:
                    if static_elements.count(map[row, col]) == 0 and map[row, col] != int(NODATA_value):
                        error_dic.update({"(%s,%s)" % (row, col): "zero but no static element and not a NODATA_value"})
                # patch element != 0
                else:
                    # check that cell is not a static element or NODATA_value
                    if static_elements.count(map[row, col]) != 0 or map[row, col] == int(NODATA_value):
                        error_dic.update({"(%s,%s)" % (row, col): "non-zero but static element or NODATA_value"})
                                       # plausibility checks are okay, then
                    else:
                        # add the land use index to the genome
                        if genom[patches[row, col] - 1] == 0:
                            genom[patches[row, col] - 1] = map[row, col]
                        # scan all patch elements
                        # check if land use is equal
                        # and mark them in the help_map
                        determine_IDmap_patch_elements(row, col, patches, map, help_map, patches[row, col],
                                                       genom[patches[row, col] - 1], error_dic, four_neighbours)
                # mark the scanned cell in help_map
                # so the cell is not scanned again by the determine_IDmap_patch_elements
                help_map[row, col] = 1


    return patches, genom.tolist()


def determine_patch_elements(row, col, map, patch_map, patch_ID, cls, four_neighbours):
    """This recursive function scans all patch elements
       and returns the coordinates of these elements.

       input:
           col and row are coordinates of the parent element
           map is the original ascii map
           patch_map is a map with patch_IDs for each patch element
           patch_ID is the ID of the new patch
           cls is the land use index of the patch
           four_neighbours if True than 4 neighboring cells are scanned else 8
    """
    # determine coordinates of neighboring cells
    new_nbhs_row, new_nbhs_col = getNbh(col, row, map.shape[1], map.shape[0], four_neighbours)
    # stack for patch elements whose neighboring cells should be determined
    nbhs_row = []
    nbhs_col = []
    for i in range(len(new_nbhs_row)):
        # add new neighboring cells to nbhs_row/col if new cells belong to cls and are not jet marked as patch element
        # the cell is no patch element if it has another land use id
        if map[new_nbhs_row[i], new_nbhs_col[i]] == cls and patch_map[new_nbhs_row[i], new_nbhs_col[i]] == 0:
            nbhs_row.append(new_nbhs_row[i])
            nbhs_col.append(new_nbhs_col[i])
    while len(nbhs_row) > 0:
        # cells could be double in nbhs_row/col
        if patch_map[nbhs_row[0], nbhs_col[0]] == 0:
            # mark all patch elements in patch_map with patch_ID
            patch_map[nbhs_row[0], nbhs_col[0]] = patch_ID
            # get coordinates of neighboring cells of this cell
            new_nbhs_row, new_nbhs_col = getNbh(nbhs_col[0], nbhs_row[0], map.shape[1], map.shape[0], four_neighbours)
            for i in range(len(new_nbhs_row)):
                # add new neighboring cells to nbhs_row/col if new cells belong to cls and are not jet marked as patch element
                if map[new_nbhs_row[i], new_nbhs_col[i]] == cls and patch_map[new_nbhs_row[i], new_nbhs_col[i]] == 0:
                    nbhs_row.append(new_nbhs_row[i])
                    nbhs_col.append(new_nbhs_col[i])
        # delete this checked neighboring cell of the array
        del nbhs_row[0]
        del nbhs_col[0]

    return patch_map

def create_patch_ID_map(map, NODATA_value, static_elements, four_neighbours):
    """This function clusters the cells of the original map into patches
       and returns a patch ID map as a 2 dimensional array and the start individual as vector.

       input:
           map is the original ascii map
           NODATA_value is the NODATA_value of the original map
           static_elements are the land use indices excluded from the optimization
           four_neighbours if True than 4 neighboring cells are scanned else 8
    """

    patches = np.zeros([map.shape[0], map.shape[1]], int)
    ids = 0
    NoData = int(NODATA_value)
    genom = []
    # loop over all cells
    for row in range(0, map.shape[0]):
        for col in range(0, map.shape[1]):
            # patchID = 0 used for static_elements
            # map element was not scanned before as patch element and is not a static element or the NODATA_value
            if patches[row, col] == 0 and static_elements.count(map[row, col]) == 0 and map[row, col] != NoData:
                cls = map[row, col]
                # increment scanned patch ID
                ids += 1
                # marke this cell as scanned patch element
                patches[row, col] = ids
                determine_patch_elements(row, col, map, patches, ids, cls, four_neighbours)
                # add the map cell value to the individual vector
                genom.append(cls)

    return patches

class Solution:
    _id = 0
    def __init__(self, representation, objective_values, landuse_order):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation
        self.objective_values = objective_values
        self.landuse_order = landuse_order

def get_optimal_solutions_GA_CoMOLA(comola_raw):
    comola_pf = []
    for i in comola_raw:
        repr = []
        obj_values = []
        for j in range(len(i)):
            if j <= 38:
                if j == 0:
                    repr.append(int(i[j][1]))
                elif j == 38:
                    repr.append(int(i[j][1]))
                else:
                    repr.append(int(i[j]))
            else:
                if j == 39:
                    obj_values.append(float(i[j][1:]))
                elif j == i.shape[0] - 1:
                    obj_values.append(float(i[j][:-1]))
                else:
                    obj_values.append(float(i[j]))
        comola_pf.append(
            Solution(representation=repr, objective_values=[obj_values[2],obj_values[0],obj_values[1],obj_values[3]], landuse_order=[1, 2, 3, 4, 5, 6, 7]))
    return comola_pf