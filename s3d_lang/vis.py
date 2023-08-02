import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.spatial.transform import Rotation as R

# Visualization logic for 3D shapes

# Draws a box (cube) on the given axes, in the given color
def draw_box(ax, cube, color):
    # calculate box attributes
    
    center = np.array([cube.X, cube.Y, cube.Z])
    lengths = np.array([cube.W, cube.H, cube.D])

    dir_1, dir_2, dir_3 = cube.getFaceNormals()
    
    rot = np.matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    center = (rot * center.reshape(-1, 1)).reshape(-1)
    dir_1 = (rot * dir_1.reshape(-1, 1)).reshape(-1)
    dir_2 = (rot * dir_2.reshape(-1, 1)).reshape(-1)
    dir_3 = (rot * dir_3.reshape(-1, 1)).reshape(-1)
    
    dir_1 = dir_1/np.linalg.norm(dir_1)
    dir_2 = dir_2/np.linalg.norm(dir_2)
    dir_3 = dir_3/np.linalg.norm(dir_3)

    # get all corners
    
    cornerpoints = np.zeros([8, 3])

    d1 = 0.5*lengths[0]*dir_1
    d2 = 0.5*lengths[1]*dir_2
    d3 = 0.5*lengths[2]*dir_3

    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3

    # plot lines
    
    ax.plot([cornerpoints[0][0], cornerpoints[1][0]], [cornerpoints[0][1], cornerpoints[1][1]],
            [cornerpoints[0][2], cornerpoints[1][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[2][0]], [cornerpoints[0][1], cornerpoints[2][1]],
            [cornerpoints[0][2], cornerpoints[2][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[3][0]], [cornerpoints[1][1], cornerpoints[3][1]],
            [cornerpoints[1][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[3][0]], [cornerpoints[2][1], cornerpoints[3][1]],
            [cornerpoints[2][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[5][0]], [cornerpoints[4][1], cornerpoints[5][1]],
            [cornerpoints[4][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[6][0]], [cornerpoints[4][1], cornerpoints[6][1]],
            [cornerpoints[4][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[5][0], cornerpoints[7][0]], [cornerpoints[5][1], cornerpoints[7][1]],
            [cornerpoints[5][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[6][0], cornerpoints[7][0]], [cornerpoints[6][1], cornerpoints[7][1]],
            [cornerpoints[6][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[4][0]], [cornerpoints[0][1], cornerpoints[4][1]],
            [cornerpoints[0][2], cornerpoints[4][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[5][0]], [cornerpoints[1][1], cornerpoints[5][1]],
             [cornerpoints[1][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[6][0]], [cornerpoints[2][1], cornerpoints[6][1]],
            [cornerpoints[2][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[3][0], cornerpoints[7][0]], [cornerpoints[3][1], cornerpoints[7][1]],
            [cornerpoints[3][2], cornerpoints[7][2]], c=color)


# Render a scene of parts, and save to name    
def scene_render(parts, name):
    fig = plt.figure()
    
    extent = 1.0

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # convert between coordinate systems
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    ax.set_proj_type('persp')
    ax.set_box_aspect(aspect = (1,1,1))
    
    ax.set_xlim(-extent, extent)
    ax.set_ylim(extent, -extent)
    ax.set_zlim(-extent, extent)

    for i, cube in enumerate(parts):
        color = (0.0, 0.0, 0.0)
        draw_box(ax, cube, color)
                   
    plt.tight_layout()

    if name is None:
        plt.show()
    else:
        plt.savefig(f'{name}.png')        
        plt.close('all')
        plt.clf()


# A dummy cube class that can be fed into the drawing code
class makeCube:
    def __init__(self, cdata):
                
        W, H, D, X, Y, Z, XA, YA, ZA = cdata
        self.X = X
        self.Y = Y
        self.Z = Z
        self.W = W
        self.H = H
        self.D = D
        
        TM = R.from_euler('xyz', [XA, YA, ZA], degrees=False).as_matrix()

        RIGHTF = np.array([1.0, 0., 0.0])
        TOPF = np.array([0.0, 1.0, 0.0])
        FRONTF = np.array([0.0, 0., 1.0])
                        
        self.Xnorm = TM @ RIGHTF
        self.Ynorm = TM @ TOPF
        self.Znorm = TM @ FRONTF
            
    def getFaceNormals(self):
        return self.Xnorm, self.Ynorm, self.Znorm

# View two sets of parts next to each other
# save result to name
def side_by_side_render(parts_a, parts_b, name):
    fig = plt.figure()
    
    extent = 1.0
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # transform between coordinate systems
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    ax.set_proj_type('persp')
    ax.set_box_aspect(aspect = (1,1,1))
    
    ax.set_xlim(-extent, extent)
    ax.set_ylim(extent, -extent)
    ax.set_zlim(-extent, extent)

    # give each set of parts a different color
    for i, cube in parts_b:
        cb = makeCube(cube)
        draw_box(ax, cb, 'black')
        
    for i, cube in parts_a:
        cb = makeCube(cube)
        draw_box(ax, cb, 'blue')

    
    # transform coordinates so z is up (from y up)
                
    plt.tight_layout()

    if name is None:
        plt.show()
    else:
        plt.savefig(f'{name}.png')        
        plt.close('all')
        plt.clf()
