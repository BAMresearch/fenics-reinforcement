# This file creates a 3D-quadrilateral mesh in the shape of a cuboid. The user has to provide two points, defined by their x,y,z coordinates, that are most far away from each other, i.e. the first point might be located at the origin (0,0,0) and the second point might be defined by (l,w,h), meaning length, width and height of the cuboid. "margin" defined as an absolute value, is the smallest distance between the outer edges of the concrete mesh and a reinforcement element. The reinforcement discretization is given by n_x and n_y, they define how many reinforcement elements should be placed in x and y direction. The parameter s defines the maximal element size in the mesh. It may be corrected according to n_x and n_y, such that the reinforcement discretization is always held true and reinorcement&concrete elements share the same nodes. "where" describes if one or two slabs of concrete should be added. Valid options are "upper", "lower" and "both". The distance between those slabs and the upper and/or lower edge of the concrete mesh is again defined by "margin". 

import gmsh
import numpy as np

def create_concrete_slab(point1: list, point2: list, n_x: int, n_y: int, margin: float, s: float, filename="test_mesh.msh", where="both"):
    
    x0,y0,z0 = point1
    l,w,h = point2
    
    # initialize gmsh
    gmsh.initialize()
    
    # alias to facilitate code writing
    mymesh = gmsh.model
    
    # create the first point
    point1 = mymesh.occ.addPoint(x0, y0, z0)
    mymesh.occ.synchronize()
    
    #  correct number of concrete elements (defined by s) in order to fit the coice of n_x and n_y
    def num_elems(min_amount_elems,reinf_elems):
        reinf_elems -= 1 
        if min_amount_elems % reinf_elems == 0:
            return min_amount_elems
        else:
            return int(np.ceil(min_amount_elems/reinf_elems))*reinf_elems
        
    concrete_elems_x = num_elems(int((l-2*margin)/s),n_x+1) 
    concrete_elems_y = num_elems(int((w-2*margin)/s),n_y+1)

    # extrude three times, point -> line (x-direction), line -> rectangle (y-direction), rectangle -> cuboid (z-direction)
    mymesh.occ.extrude([(0, point1)],l,0,0, numElements=[1,concrete_elems_x,1], heights=[margin/l,1-margin/l,1], recombine=True)
    mymesh.occ.synchronize()
    
    
    mymesh.occ.extrude([(1, 1)],0,w,0, numElements=[1,concrete_elems_y,1], heights=[margin/w,1-margin/w,1], recombine=True)
    mymesh.occ.synchronize()
    
    mymesh.occ.extrude([(2, 1)],0,0,h, numElements=[1,int((h-2*margin)/s),1], heights=[margin/h,1-margin/h,1], recombine=True)
    mymesh.occ.synchronize()
    
    
    # generate the 3D-mesh as dictated in extrude
    meshFact = gmsh.model.mesh
    meshFact.generate(3)
    

    
    # add points and lines where reinforcement is to be added (depending on n_x and n_y), function is called for x and y separately
    x = [x0+margin,l-margin]
    y = [y0+margin,w-margin]
    def reinforcement_points(Nx,x,y,addx,where,i):
        if where=="upper":
            z = [1-margin]
        elif where=="lower":
            z = [margin]
        elif where=="both":
            z = [margin,1-margin]
            
        reinf_tags = [] # save tags to create lines later
        ntp = mymesh.occ.getEntities(0)[-1][-1]+1 # "next tag point"
        x_cords = np.linspace(x[0],x[-1],Nx) # contains all x (or y) coordinates where points have to be added
        x_cords = np.repeat(x_cords,2) # doubles every element, because the same x value is added on two y values
        y_cords = np.array(y*Nx) # contains the two y - limits as often as necessary to add all points
           
        for z_cord in z:
            for (x_cord,y_cord) in zip(x_cords,y_cords):
                i += 1
                
                # check if points for horizontal (x) or vertical (y) lines should be added
                if addx == True: 
                    mymesh.occ.addPoint(x_cord, y_cord, z_cord, tag = ntp+i) 
                    
                else:
                    mymesh.occ.addPoint(y_cord, x_cord, z_cord, tag = ntp+i) 
                reinf_tags.append(ntp+i)
        return reinf_tags,i
    
    i_points = 0
    reinf_tags_x,i_points = reinforcement_points(n_x+1, x, y,True,where,i_points) # for reinforcement in x-direction
    
    # before adding reinforcement in y-direction, it is necessary to know which was the last point that has been added before
    ltp_Nx = mymesh.occ.getEntities(0)[-1][-1] # "last tag point"
    reinf_tags_y,i_points = (reinforcement_points(n_y+1,y,x,False,where,i_points)) # for y-reinforcement
    reinf_tags = np.hstack((reinf_tags_x,reinf_tags_y)) # save all reinforcement tags 
    
    # add lines which connect the reinforcement points
    def reinforcement_lines(reinf_tags,ltp_Nx):
        line_tags = [] # save line tags for physical group
        ltl = mymesh.occ.getEntities(1)[-1][-1] # "last tag line"
        k=0
        
        for i,point_tag in enumerate(reinf_tags):
            if i%2 != 0 or i == ltp_Nx+1: # otherwise you will connect all points twice
                k+=1 # to find the correct tags
                mymesh.occ.addLine(reinf_tags[i-1], point_tag, tag=ltl+k) 
                line_tags.append(ltl+k)
        return line_tags
    
    line_tags = reinforcement_lines(reinf_tags,ltp_Nx) 
    
    # add new points to mesh by synchronizing
    mymesh.occ.synchronize() 
    
    # add all reinforcement lines to a physical group
    mymesh.addPhysicalGroup(dim = 1, tags = line_tags, tag = 1)
    
    # Save mesh as msh file
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(filename)
    gmsh.finalize



    










