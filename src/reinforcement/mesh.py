# This file creates a 3D-quadrilateral mesh. The user has to provide two points, defined by their x,y,z coordinates, that are most far away from each other, i.e. the first point might be the origin (0,0,0) and the second point might be defined by (l,w,h), meaning length, width and height. "margin" defines the distance between the margin of the concrete mesh and the reinforcement bar next to it, which is going to be implemented in the future. 

# import 
import gmsh






def create_concrete_slab(point1: list, point2: list, n_x: int, n_y: int, margin: float, filename="test_mesh.msh", where="both"):
    
    x0,y0,z0 = point1
    l,w,h = point2
    gmsh.initialize()
    
    # alias to facilitate code writing
    mymesh = gmsh.model
    point1 = mymesh.occ.addPoint(x0, y0, z0)
    point2 = mymesh.occ.addPoint(l, y0, z0)
    point3 = mymesh.occ.addPoint(l, w, z0)
    point4 = mymesh.occ.addPoint(x0, w, z0)
    mymesh.occ.synchronize()
    
    line1 = mymesh.occ.addLine(point1, point2)
    line2 = mymesh.occ.addLine(point2, point3)
    line3 = mymesh.occ.addLine(point3, point4)
    line4 = mymesh.occ.addLine(point4, point1)
    
    face1 = mymesh.occ.addCurveLoop([line1, line2, line3, line4])
    mymesh.occ.synchronize()
    
    mymesh.occ.addPlaneSurface([face1])
    mymesh.occ.synchronize()
    mymesh.addPhysicalGroup(dim = 2, tags = [face1]) 
    
    # Meshing
    meshFact = gmsh.model.mesh

    # transfinite curves
    n_nodes = 10

    meshFact.setTransfiniteCurve(line1, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(line2, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(line3, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(line4, numNodes=n_nodes)

    # transfinite surfaces
    meshFact.setTransfiniteSurface(face1)
    mymesh.mesh.setRecombine(2,face1,angle=90)

    # generate a mesh on the base surface (2D)
    meshFact.generate(2)
    
    dz = h
    mymesh.occ.extrude([(2, face1)], 0., 0., dz,
                    numElements=[1,1,1], heights=[0.1,0.9,1], recombine=True)
    mymesh.occ.synchronize()

    # generate a mesh on the whole geometry (3D)
    meshFact.generate(3)
    gmsh.fltk.run() # TODO no graphical user interface
    # Write mesh data:
    gmsh.write(filename)
    gmsh.finalize()




if __name__=="__main__":
    # user input (example)
    l = 0.5 # length 2.75
    w = 0.4 # width
    h = 0.2 # height 
    point1 = [0,0,0]
    point2 = [l,w,h]
    filename = "test_mesh.msh"
    create_concrete_slab(point1=point1, point2=point2, n_x=0, n_y=0, margin=0.1, where="both")










