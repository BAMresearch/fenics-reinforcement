# This file creates a 3D-quadrilateral mesh. The user has to provide two points, defined by their x,y,z coordinates, that are most far away from each other, i.e. the first point might be the origin (0,0,0) and the second point might be defined by (l,w,h), meaning length, width and height. "margin" defines the distance between the margin of the concrete mesh and the reinforcement bar next to it, which is going to be implemented in the future. 

# import 
import gmsh
import numpy as np







def create_concrete_slab(point1: list, point2: list, n_x: int, n_y: int, margin: float, s: float, filename="test_mesh.msh", where="both"):
    
    x0,y0,z0 = point1
    l,w,h = point2
    n_x += 1 # nx is the amount of elements, that means one extra line has to be added
    n_y += 1 # same as nx
    gmsh.initialize()
    
    # alias to facilitate code writing
    mymesh = gmsh.model
    
    point1 = mymesh.occ.addPoint(x0, y0, z0)
    mymesh.occ.synchronize()
    
    # add as many elements in order to reach an element size of s or smaller. The element size also has to match your choice of n_x and n_y, so that the reinforcement lines and the lines of the concrete mesh are to be found on top of each other (reinforcement and concrete share their nodes)
    def num_elems(analyt,direct):
        direct = direct-1
        if analyt % direct == 0:
            return analyt
        else:
            return int(np.ceil(analyt/direct))*direct
        
    ax = num_elems(int((l-2*margin)/s),n_x) 
    ay = num_elems(int((w-2*margin)/s),n_y)
    

    
    mymesh.occ.extrude([(0, point1)],l,0,0, numElements=[1,ax,1], heights=[margin/l,1-margin/l,1], recombine=True)
    mymesh.occ.synchronize()
    
    
    mymesh.occ.extrude([(1, 1)],0,w,0, numElements=[1,ay,1], heights=[margin/w,1-margin/w,1], recombine=True)
    mymesh.occ.synchronize()
    
    mymesh.occ.extrude([(2, 1)],0,0,h, numElements=[1,int((h-2*margin)/s),1], heights=[margin/h,1-margin/h,1], recombine=True)
    mymesh.occ.synchronize()
    
    
    # Meshing
    meshFact = gmsh.model.mesh
    meshFact.generate(3)
    
    # hier dann Anzahl nx bzw ny verwenden um neue reinforcement Punkte hinzuzufügen (siehe add reinf Funktion aus alter Datei), dann wieder Physical Curves usw
    
    # function that adds points and lines where reinforcement is to be added (depending on n_x and n_y)
    # in order to create lines which can be added to a physical group later,
    # we first need to define points, which are then interconnected. Their tags
    # are saved so that they can be saved as a physical group later
    # the following function is called twice, once for x and then for y
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
                
                # check if points for horizontal or vertical lines should be added
                if addx == True: 
                    mymesh.occ.addPoint(x_cord, y_cord, z_cord, tag = ntp+i) 
                    
                else:
                    mymesh.occ.addPoint(y_cord, x_cord, z_cord, tag = ntp+i) 
                reinf_tags.append(ntp+i)
        return reinf_tags,i
    
    i_points = 0
    reinf_tags_x,i_points = reinforcement_points(n_x, x, y,True,where,i_points) # for Nx refinement
    # before adding Ny refinement it is necessary to know which was the last point
    # added -> will be used in reinforcement_lines to make sure the correct points
    # are conncted to each other
    ltp_Nx = mymesh.occ.getEntities(0)[-1][-1] # "last tag point"
    reinf_tags_y,i_points = (reinforcement_points(n_y,y,x,False,where,i_points)) # for Ny refinement
    reinf_tags = np.hstack((reinf_tags_x,reinf_tags_y)) # save all reinforcement tags together in one array
    
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
    
    # add our new points to our mesh by synchronizing
    mymesh.occ.synchronize() 
    
    # add all reinforcement lines to a physical group
    mymesh.addPhysicalGroup(dim = 1, tags = line_tags, tag = 1)
    
    mymesh.occ.synchronize() 
  



    # Save mesh as msh file
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(filename)
    gmsh.finalize

if __name__=="__main__":
    
    # user input (example)
    l = 1.5 # length 2.75
    w = 2 # width
    h = 1 # height 
    point1 = [0,0,0]
    point2 = [l,w,h]
    margin=0.25
    nx = 13 # reinforcement density (number of elements) in x direction
    ny=7 # reinforcement density (number of elements) in y direction
    s_exp = 0.25
    filename = "test_mesh.msh"
    create_concrete_slab(point1=point1, point2=point2, n_x=nx, n_y=ny, margin=margin, s=s_exp, where="upper")
    










