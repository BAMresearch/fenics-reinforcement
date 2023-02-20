# This file covers all functions related to the mesh generation in gmsh and its conversion into separated xdmf files, which can then be used in dolfinx.

import gmsh
import numpy as np
import meshio
import dolfinx as dfx
from mpi4py import MPI
from numpy.typing import ArrayLike
import math
from itertools import product

def _num_elems(min_amount_elems, reinf_elems):
    '''
    Corrects number of concrete elements (defined by s) in order to fit the coice of n_x and n_y.
    '''
    if reinf_elems == 0:
        return min_amount_elems
    elif min_amount_elems % reinf_elems == 0:
        return min_amount_elems
    else:
        return int(np.ceil(min_amount_elems / reinf_elems)) * reinf_elems
    
def _reinforcement_points(Nx, Ny, x, y, addx, z, margin):
    '''
    Generates reinforcement nodes.
    ''' 

    
    i = 0
    reinf_tags = []  # save tags to create lines later
    reinf_tags_2 = [] # create a second list for the case "both" (makes creating lines later easier)
    ntp = gmsh.model.occ.getEntities(0)[-1][-1] + 1  # "next tag point"
    x_cords = np.linspace(
        x[0], x[-1], Nx 
    )  # contains all x coordinates where points have to be added (lines in y direction)

    y_cords = np.linspace(
        y[0], y[-1], len(y)
    )  # contains all y coordinates where points have to be added (lines in y direction)

    x_cords_2 = np.linspace(x[0], x[-1], len(x)) # add additional x coordinates for lines in x direction
    y_cords_2 = np.linspace(y[0],y[-1],Ny) # additional y coordinates for lines in x direction
    
   
    # TODO fix x_cord_2, y_cord_2 -> all in one loop or two separate loops?
    for z_cord in z:
       for x_cord in x_cords:
           for y_cord in y_cords:
               i += 1
               gmsh.model.occ.addPoint(x_cord, y_cord, z_cord, tag=ntp + i)
               if z_cord == z[0]:
                   reinf_tags.append(ntp + i)
               else:
                   reinf_tags_2.append(ntp+i)
       for y_cord in y_cords_2:
           for x_cord in x_cords_2:
               i+=1
               gmsh.model.occ.addPoint(x_cord, y_cord, z_cord, tag=ntp + i)
               if z_cord == z[0]:
                   reinf_tags.append(ntp + i)
               else:
                   reinf_tags_2.append(ntp+i)

            
    return reinf_tags, reinf_tags_2


def _reinforcement_lines(reinf_tags,reinf_tags_2,Nx,Ny,elems_x,elems_y):
    '''
    Generates lines that connect the reinforcement nodes.
    '''
    line_tags = []  # save line tags for physical group
    ltl = gmsh.model.occ.getEntities(1)[-1][-1]  # "last tag line"
    k_tags = 0 # to find the correct tags
    k_x_or_y = 0 # keep track whether lines are added in x or y direction
    counter_y = 0
    counter_x = 0
    
    def _create_lines_from_tags(reinf_tags,line_tags,ltl,k_tags,k_x_or_y,counter_y,counter_x):
        for i, point_tag in enumerate(reinf_tags):
            k_tags += 1  
            k_x_or_y += 1 
            counter_y += 1
            
            if k_x_or_y<len(reinf_tags) - (elems_x+1)*Ny: # create lines in y direction
                if counter_y == elems_y+1:
                    counter_y = 0 # do not add line and reset counter
                    
                else:
                    gmsh.model.occ.addLine(point_tag, reinf_tags[i + 1], tag=ltl + k_tags)
                    line_tags.append(ltl + k_tags)
            elif k_x_or_y<len(reinf_tags) and k_x_or_y > len(reinf_tags) - (elems_x+1)*Ny: # create lines in x direction
                counter_x += 1
               
                if counter_x == elems_x+1:
                    counter_x = 0
                else:
                    gmsh.model.occ.addLine(point_tag, reinf_tags[i + 1], tag=ltl + k_tags)
                    line_tags.append(ltl + k_tags)
        return ltl,k_tags,line_tags 
    
    ltl,k_tags,line_tags = _create_lines_from_tags(reinf_tags, line_tags, ltl, k_tags,k_x_or_y, counter_y, counter_x)
   
    if len(reinf_tags_2)>0:
       
        _,_,line_tags = _create_lines_from_tags(reinf_tags_2, line_tags, ltl, k_tags,k_x_or_y, counter_y, counter_x)
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
    z: list,
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
    z: list
        Position of the reinforecement bar(s). The list may contain either one or two floats, dictating whether one or two reinforcement bars should be created. The float represents the z-coordinate of the bar(s)

    Returns
    -------

    """
   
    x0, y0, z0 = point1
    l, w, h = point2

    # initialize gmsh
    gmsh.initialize()

    # alias to facilitate code writing
    mymesh = gmsh.model
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",100000)
    # create the first point
    point1 = mymesh.occ.addPoint(x0, y0, z0)
    mymesh.occ.synchronize()

    concrete_elems_x = _num_elems(int((l - 2 * margin) / s), n_x-1)
    concrete_elems_y = _num_elems(int((w - 2 * margin) / s), n_y-1)
    n_elements_margin = math.ceil(margin/s)
    
    # evaluate z, find correct margins and amount of elements in z-direction
    if len(z) == 2 :
        n_elements_margin_z_1 = math.ceil(z[0]/s)
        n_elements_margin_z_2 = math.ceil((h-z[1])/s)
        heights_z = [z[0]/h, z[1]/h, 1]
        numElements_z = [n_elements_margin_z_1, int((h - (z[0]+(h-z[1]))) / s), n_elements_margin_z_2]
        
    elif len(z) == 1:
        n_elements_margin_z_1 = math.ceil(z[0]/s)
        n_elements_margin_z_2 = math.ceil((h-z[0])/s)
        heights_z = [z[0]/h,1]
        numElements_z = [n_elements_margin_z_1,n_elements_margin_z_2]
        
        
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
        numElements=numElements_z,
        heights=heights_z,
        recombine=True,
    ) 
    mymesh.occ.synchronize()

    
    # add volume as physical group
    mymesh.addPhysicalGroup(dim=3, tags=[1], tag=1)
    mymesh.occ.synchronize()

    # add points and lines where reinforcement is to be added 
    x = np.linspace(x0+margin, l-margin,concrete_elems_x+1) 
    y = np.linspace(y0+margin, w-margin,concrete_elems_y+1)
    reinf_tags, reinf_tags_2 = _reinforcement_points(
        n_x, n_y, x, y, True, z, margin
    )  

    # add lines connecting all reinforcement nodes and save their tags
    line_tags = _reinforcement_lines(reinf_tags,reinf_tags_2,n_x,n_y,concrete_elems_x,concrete_elems_y) 
    
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
