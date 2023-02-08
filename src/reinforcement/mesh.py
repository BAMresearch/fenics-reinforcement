# This file covers all functions related to the mesh generation in gmsh and its conversion into separated xdmf files, which can then be used in dolfinx.

import gmsh
import numpy as np
import meshio
import dolfinx as dfx
from mpi4py import MPI
from numpy.typing import ArrayLike
import math


def _num_elems(min_amount_elems, reinf_elems):
    '''
    Corrects number of concrete elements (defined by s) in order to fit the coice of n_x and n_y.
    '''
    reinf_elems -= 1
    if min_amount_elems % reinf_elems == 0:
        return min_amount_elems
    else:
        return int(np.ceil(min_amount_elems / reinf_elems)) * reinf_elems
    
def _reinforcement_points(Nx, x, y, addx, where, i, margin):
    '''
    Generates reinforcement nodes.
    '''
    if where == "upper":
        z = [1 - margin]
    elif where == "lower":
        z = [margin]
    elif where == "both":
        z = [margin, 1 - margin]

    reinf_tags = []  # save tags to create lines later
    ntp = gmsh.model.occ.getEntities(0)[-1][-1] + 1  # "next tag point"
    x_cords = np.linspace(
        x[0], x[-1], Nx
    )  # contains all x (or y) coordinates where points have to be added
    x_cords = np.repeat(
        x_cords, 2
    )  # doubles every element, because the same x value is added on two y values
    y_cords = np.array(
        y * Nx
    )  # contains the two y - limits as often as necessary to add all points

    for z_cord in z:
        for (x_cord, y_cord) in zip(x_cords, y_cords):
            i += 1
            print("ntp+i", ntp + i)
            # check if points for horizontal (x) or vertical (y) lines should be added
            if addx == True:
                gmsh.model.occ.addPoint(x_cord, y_cord, z_cord, tag=ntp + i)

            else:
                gmsh.model.occ.addPoint(y_cord, x_cord, z_cord, tag=ntp + i)
            reinf_tags.append(ntp + i)
    return reinf_tags, i


def _reinforcement_lines(reinf_tags):
    '''
    Generates lines that connect the reinforcement nodes.
    '''
    line_tags = []  # save line tags for physical group
    ltl = gmsh.model.occ.getEntities(1)[-1][-1]  # "last tag line"
    k = 0

    for i, point_tag in enumerate(reinf_tags):
        if i % 2 != 0:  # otherwise you will connect all points twice
            k += 1  # to find the correct tags
            gmsh.model.occ.addLine(reinf_tags[i - 1], point_tag, tag=ltl + k)
            line_tags.append(ltl + k)
    return line_tags


def _create_xdmf(msh_file,xdmf_files):
    '''
    Reads mesh (msh file) and creates two xdmf files (conrete & reinforcement) from it.
    '''
    def create_mesh(mesh, cell_type):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points
        out_mesh = meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data]},
        )
        return out_mesh

    # Read mesh
    msh = meshio.read(msh_file)

    # Create and save one file containing the whole volume (hexahedra), and one file for the reinforcement lines
    hexa_mesh = create_mesh(msh, "hexahedron")
    line_mesh = create_mesh(msh, "line")

    meshio.write(xdmf_files[0], hexa_mesh)
    meshio.write(xdmf_files[1], line_mesh)

def create_concrete_slab(
    point1: ArrayLike,
    point2: ArrayLike,
    n_x: int,
    n_y: int,
    margin: float,
    s: float,
    msh_filename: str,
    xdmf_filenames: list,
    where="both",
):
    """
    This function creates a 3D-quadrilateral mesh in the shape of a cuboid with reinforcement
    rebars as line elements in a 3D space. Only a very general case with a uniform rebar-grid
    at the top and/or bottom of the slab is possible.

    Parameters
    ----------
    point1: ArrayLike
        Starting point of the cuboid.
    point2: ArrayLike
        Endpoint of the cuboid.
    n_x: int
        Number of reinforcement bars in x-direction. 
    n_y: int
        Number of reinforcement bars in y-direction.
    margin: float
        Absolute value of the smallest distance between the outer edges of the concrete mesh
        and a reinforcement element.
    s: float
        Maximal element size. It may be corrected according to n_x and n_y, such that
        the reinforcement discretization is always held true and reinorcement&concrete
        elements share the same nodes.
    msh_filename: str
        Filename of the mesh (msh file).
    xdmf_filenames : list
        Desired names of the two xdmf meshes (concrete & reinforcement).
    where: str
        Position of the reinforecement bars. Valid options are "upper", "lower"
        and "both".

    Returns
    -------

    """

    x0, y0, z0 = point1
    l, w, h = point2

    # initialize gmsh
    gmsh.initialize()

    # alias to facilitate code writing
    mymesh = gmsh.model

    # create the first point
    point1 = mymesh.occ.addPoint(x0, y0, z0)
    mymesh.occ.synchronize()

    concrete_elems_x = _num_elems(int((l - 2 * margin) / s), n_x + 1)
    concrete_elems_y = _num_elems(int((w - 2 * margin) / s), n_y + 1)
    n_elements_margin = math.ceil(margin/s)
    
    # extrude three times, point -> line (x-direction), line -> rectangle (y-direction), rectangle -> cuboid (z-direction)
    mymesh.occ.extrude(
        [(0, point1)],
        l,
        0,
        0,
        numElements=[n_elements_margin, concrete_elems_x, n_elements_margin],
        heights=[margin / l, 1 - margin / l, 1],
        recombine=True,
    )
    mymesh.occ.synchronize()

    mymesh.occ.extrude(
        [(1, 1)],
        0,
        w,
        0,
        numElements=[n_elements_margin, concrete_elems_y, n_elements_margin],
        heights=[margin / w, 1 - margin / w, 1],
        recombine=True,
    )
    mymesh.occ.synchronize()

    mymesh.occ.extrude(
        [(2, 1)],
        0,
        0,
        h,
        numElements=[n_elements_margin, int((h - 2 * margin) / s), n_elements_margin],
        heights=[margin / h, 1 - margin / h, 1],
        recombine=True,
    )
    mymesh.occ.synchronize()

    # add volume as physical group
    mymesh.addPhysicalGroup(dim=3, tags=[1], tag=1)
    mymesh.occ.synchronize()

    # add points and lines where reinforcement is to be added (depending on n_x and n_y), function is called for x and y separately
    x = [x0 + margin, l - margin]
    y = [y0 + margin, w - margin]
    i_points = 0
    
    reinf_tags_x, i_points = _reinforcement_points(
        n_x + 1, x, y, True, where, i_points, margin
    )  # for reinforcement in x-direction

    reinf_tags_y, i_points = _reinforcement_points(
        n_y + 1, y, x, False, where, i_points, margin
    )  # for y-reinforcement
    reinf_tags = np.hstack((reinf_tags_x, reinf_tags_y))  # save all reinforcement tags
    line_tags = _reinforcement_lines(reinf_tags)

    # add new points to mesh by synchronizing
    mymesh.occ.synchronize()

    # add all reinforcement lines to a physical group
    mymesh.addPhysicalGroup(dim=1, tags=line_tags, tag=1)
    mymesh.occ.synchronize()

    # now generate the 3D-mesh
    meshFact = gmsh.model.mesh
    meshFact.generate(3)

    # Save mesh as msh file
    gmsh.write(msh_filename)
    gmsh.finalize
    
    # create two xdmf files (concrete & reinforcement)
    _create_xdmf(msh_filename,xdmf_filenames)


def read_xdmf(xdmf_files):
    '''
    File that reads xdmf_files to use them in FEniCSx

    Parameters
    ----------
    xdmf_files : list
        Names (str) of the xdmf_files, [concrete, reinforcement] - in this order.

    Returns
    -------
    concrete_mesh : dolfinx.mesh.Mesh
        The concrete mesh (hexa elements).
    rebar_mesh : dolfinx.mesh.Mesh
        The reinforcement mesh (line elements).

    '''
    with dfx.io.XDMFFile(MPI.COMM_WORLD, xdmf_files[0], "r") as xdmf:
        concrete_mesh = xdmf.read_mesh(name="Grid")

    with dfx.io.XDMFFile(MPI.COMM_WORLD, xdmf_files[1], "r") as xdmf:
        rebar_mesh = xdmf.read_mesh(name="Grid")
    return concrete_mesh, rebar_mesh
