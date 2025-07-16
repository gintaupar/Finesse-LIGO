# Madison Simmonds, June 2024

# This uses https://github.com/danielbrown2/raypier_optics code for analysis,
# instructions for install in G2401138, under "Dan's Raypier Fork"

# Demonstrating how to apply custom surface profile to an optic by specifying sequence of Zernike polynomials to distort standard profile
# Zernike terms are identified using the ANSI standard single-index notation (J)
# Amplitude coefficients give the RMS deviation of the surface over the unit disk

# %%
import matplotlib.pyplot as plt
from raypier.api import RayTraceModel, GeneralLens, ParallelRaySource, SphericalFace, CircleShape, OpticalMaterial, HexagonalRayFieldSource, ConicFace, PlanarFace, PlanoConvexLens, RayCapturePlane, DistortionFace
from raypier.faces import BaseFace
from raypier.distortions import ZernikeSeries,SimpleTestZernikeJ7
from raypier.mirrors import PECMirror
from raypier.parabolics import OffAxisParabloid
import numpy as np


# %%
# Functions from Cao, used to create optic directions from coordinates
def target_mirror_direction(target_ray_direction, init_ray_direction):
    a_mag = np.linalg.norm(init_ray_direction)
    a= -init_ray_direction
    b_mag = np.linalg.norm(target_ray_direction)
    b = target_ray_direction
    c = a_mag * b + b_mag * a
    c /= np.linalg.norm(c)
    return tuple(c)

def mirror_direction_from_coords(src_loc, dest_loc, mirror_loc):
    """
    Compute mirror direction from coordinate of optics

    Params:
    ------------------
    src_loc: coordinate of first mirror
    dest_loc: coordinate of target mirror
    mirror_loc: coordinate of current mirror

    Returns:
    ----------------------
    mirror_dir: direction of current mirror
    """
    in_v = np.array(mirror_loc) - np.array(src_loc)
    out_v = np.array(dest_loc) - np.array(mirror_loc)
    mirror_dir = target_mirror_direction(out_v, in_v)
    return mirror_dir


def normDir(init_ray_direction):
    norm_dir_ray = init_ray_direction/np.linalg.norm(init_ray_direction)
    return norm_dir_ray

from_mm=1e-1

# %% Setting location and direction for optics

input_loc=(0,0,0)

mag = 20
f1_m = 3 #m
f1 = f1_m*1e2 #cm
ray_len = 2*f1 

input_to_M1_dist = f1 #cm


input_to_M1_ray_dir = 0 #deg
M1_to_view_ray_dir = 0 #deg


input_target_dir = (np.cos(input_to_M1_ray_dir),np.sin(input_to_M1_ray_dir),0)
M1_target_dir = (-np.cos(M1_to_view_ray_dir),np.sin(M1_to_view_ray_dir),0)


M1_dir = tuple(target_mirror_direction(np.array(M1_target_dir),np.array(input_target_dir)))

M1_loc = tuple(np.array(input_loc) + input_to_M1_dist*(np.array(input_target_dir)/np.linalg.norm(np.array(input_target_dir))))


plt.plot(input_loc[0],input_loc[1],'.')
plt.quiver(input_loc[0],input_loc[1],input_target_dir[0],input_target_dir[1])
plt.plot(M1_loc[0],M1_loc[1],'.')
plt.quiver(M1_loc[0],M1_loc[1],M1_target_dir[0],M1_target_dir[1])
plt.title("Optic location and direction")
plt.show()

# %%
### Build a mirrors/lenses ###
mtest_shape = CircleShape(radius=60/2)
base_surface = PlanarFace()


distortion_test1 = SimpleTestZernikeJ7(unit_radius=10.0,amplitude=0.02) # Test class, J=7 refers to a coma distortion
mtest1f = DistortionFace(base_face = base_surface, distortion = distortion_test1, mirror=True)
Mtest1 = GeneralLens(name="Mtest1",
                 shape=mtest_shape,
                 surfaces=[mtest1f],
                 centre=M1_loc,
                 direction=M1_dir
                 )

distortion_test2 = ZernikeSeries(unit_radius=10.0,coefficients=[(9,0.001)])
mtest2f = DistortionFace(base_face = base_surface, distortion = distortion_test2, mirror=True)
Mtest2 = GeneralLens(name="Mtest2",
                 shape=mtest_shape,
                 surfaces=[mtest2f],
                 centre=M1_loc,
                 direction=M1_dir
                 )



### Add a source ###
src = HexagonalRayFieldSource(origin=input_loc, 
                              direction= input_target_dir, 
                              gauss_width=5.0,
                              display="wires",
                              opacity=0.1,
                              radius=20/2,
                              resolution = 10,
                              show_normals=True,
                              max_ray_len = ray_len
                              )



### Add probes ###

# Adding in view plane to observe rays at arbitary location within system
view_probe = RayCapturePlane(centre = input_loc,
                           direction = input_target_dir,
                           width = 30,
                           height=30)




### Build model ###
model = RayTraceModel(optics=[
                            Mtest1,
                              # Mtest2
                              ],
                        sources=[src],
                        probes=[view_probe])


# %%
###Now open the GUI###
model.configure_traits()

# %%
M_trace = src.traced_rays[0]
term_trace = src.traced_rays[1]
view_trace = view_probe.captured[0].termination
view_dir = view_probe.captured[0].direction

# %% Analyse system


plt.plot(M_trace.termination[:,2],M_trace.termination[:,1],'.')
plt.title("Test surface")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(view_trace[:,2],view_trace[:,1],'.')
plt.title("View plane")
plt.gca().set_aspect("equal")
plt.show()

# plt.plot(view_trace.termination[:,2],view_trace.termination[:,1],'.')
# plt.title("view")
# plt.gca().set_aspect("equal")
# plt.show()



# %% View ray termination - z-y plane

plt.plot(M_trace.termination[:,0]
        ,M_trace.termination[:,1]
         ,'.')
plt.plot(M1_loc[0],M1_loc[1],'.')
plt.title("M1")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(view_trace[:,0]
        ,view_trace[:,1]
         ,'.')
# plt.plot(M2_loc[0],M2_loc[1],'.')
plt.title("view")
plt.gca().set_aspect("equal")
plt.show()

# plt.plot(view_trace.termination[:,0],view_trace.termination[:,1],'.')
# plt.plot(view_loc[0],view_loc[1])
# plt.title("view")
# plt.gca().set_aspect("equal")
# plt.show()




# %% Direction and position of rays at optics
# plt.quiver(ETM_trace.termination[:,2],ETM_trace.termination[:,1],ETM_trace.direction[:,2],ETM_trace.direction[:,1])
# plt.plot(ETM_trace.termination[:,2],ETM_trace.termination[:,1],'.')
# plt.title("ETM")
# plt.gca().set_aspect("equal")
# plt.show()

plt.quiver(M_trace.termination[:,2],M_trace.termination[:,1],M_trace.direction[:,2],M_trace.direction[:,1])
plt.plot(M_trace.termination[:,2],M_trace.termination[:,1],'.')
plt.title("M1")
plt.gca().set_aspect("equal")
plt.show()


plt.quiver(view_trace[:,2],view_trace[:,1],view_dir[:,2],view_dir[:,1])
plt.plot(view_trace[:,2],view_trace[:,1],'.')
plt.title("view")
plt.gca().set_aspect("equal")
plt.show()

# plt.quiver(view_trace.termination[:,2],view_trace.termination[:,1],view_trace.direction[:,2],view_trace.direction[:,1])
# plt.plot(view_trace.termination[:,2],view_trace.termination[:,1],'.')
# plt.title("view")
# plt.gca().set_aspect("equal")
# plt.show()




# %%
###Now open the GUI###
model.configure_traits()

# %%
