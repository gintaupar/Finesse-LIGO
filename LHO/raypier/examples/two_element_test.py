# Madison Simmonds, 2024

# This uses https://github.com/danielbrown2/raypier_optics code for analysis,
# instructions for install in G2401138, under "Dan's Raypier Fork"

# Script demonstrating a two-element optical system which adheres to the B=C=0 imaging relay condition
# Select collimated input, confocal input, or single-pass input to observe behaviour of system for a double pass system with collimated input rays or confocal input rays 

# %%
import matplotlib.pyplot as plt
from raypier.api import RayTraceModel, GeneralLens, ParallelRaySource, SphericalFace, CircleShape, OpticalMaterial, HexagonalRayFieldSource, ConicFace, PlanarFace, PlanoConvexLens, RayCapturePlane, DistortionFace,ConfocalRayFieldSource
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

from_mm=1e-1

def normDir(init_ray_direction):
    norm_dir_ray = init_ray_direction/np.linalg.norm(init_ray_direction)
    return norm_dir_ray

m = OpticalMaterial(glass_name="N-BK7")
# %%
input_loc=(0,0,0)

mag = 20
f1_m = 3 #m
f1 = f1_m*1e2 #cm
R1 = 2*f1
f2 = mag/f1_m #m
f2 = f2*1e2 #cm
R2 = 2*f2
ray_len = 2*f1 + 2*f2

input_to_M1_dist = f1 #cm
M1_to_M2_dist = f1 + f2 #cm
M2_to_TM_dist = f2 #cm

input_to_M1_ray_dir = 0 #deg
M1_to_M2_ray_dir = np.deg2rad(10) #deg
M2_to_TM_ray_dir = 0 #deg

input_target_dir = (np.cos(input_to_M1_ray_dir),np.sin(input_to_M1_ray_dir),0)
M1_target_dir = (-np.cos(M1_to_M2_ray_dir),np.sin(M1_to_M2_ray_dir),0)
M2_target_dir = (np.cos(M2_to_TM_ray_dir),np.sin(M2_to_TM_ray_dir),0)


input_dir = tuple(input_target_dir)
M1_dir = tuple(target_mirror_direction(np.array(M1_target_dir),np.array(input_dir)))
M2_dir = tuple(target_mirror_direction(np.array(M2_target_dir),np.array(M1_target_dir)))


M1_loc = tuple(np.array(input_loc) + input_to_M1_dist*(np.array(input_target_dir)/np.linalg.norm(np.array(input_target_dir))))
M2_loc = tuple(np.array(M1_loc) + M1_to_M2_dist*(np.array(M1_target_dir)/np.linalg.norm(np.array(M1_target_dir))))
TM_loc = tuple(np.array(M2_loc) + M2_to_TM_dist*(np.array(M2_target_dir)/np.linalg.norm(np.array(M2_target_dir))))

TM_dir = tuple(mirror_direction_from_coords(M2_loc, M2_loc, TM_loc))

plt.plot(input_loc[0],input_loc[1],'.')
plt.quiver(input_loc[0],input_loc[1],input_target_dir[0],input_target_dir[1])
plt.plot(M1_loc[0],M1_loc[1],'.')
plt.quiver(M1_loc[0],M1_loc[1],M1_target_dir[0],M1_target_dir[1])
plt.plot(M2_loc[0],M2_loc[1],'.')
plt.quiver(M2_loc[0],M2_loc[1],M2_target_dir[0],M2_target_dir[1])
plt.plot(TM_loc[0],TM_loc[1],'.')
plt.quiver(TM_loc[0],TM_loc[1],TM_dir[0],TM_dir[1])
plt.show()

# %%

### Build a mirrors/lenses ###

TM_shape = CircleShape(radius=60/2)
TMf = PlanarFace(mirror=True) 
TM = GeneralLens(name="TM",
                shape=TM_shape,
                surfaces=[TMf],
                centre=TM_loc,
                direction = TM_dir,
                )

m2_shape = CircleShape(radius=60/2)
m2f = ConicFace(curvature = -R2, conic_const=-1.0000000001, mirror=True)
M2 = GeneralLens(name="M2",
                 shape=m2_shape,
                 surfaces=[m2f],
                 centre=M2_loc,
                 direction=M2_dir)


m1_shape = CircleShape(radius=60/2)
m1f = ConicFace(curvature = -R1, conic_const=-1.0000000001, mirror=True)
M1 = GeneralLens(name="M1",
                 shape=m1_shape,
                 surfaces=[m1f],
                 centre=M1_loc,
                 direction=M1_dir)


# View as transparent surface
view_shape = CircleShape(radius=20/2)
viewf = PlanarFace(mirror=False) 
view = GeneralLens(name="view",
                shape=view_shape,
                surfaces=[viewf],
                centre = input_loc,
                direction = input_dir,
                )


### Add a source ###
src_collimated = HexagonalRayFieldSource(name = "src_collimated",
                              origin=input_loc, 
                              direction= input_dir, 
                              gauss_width=5.0,
                              display="wires",
                              opacity=0.1,
                              radius=20/2,
                              resolution = 10,
                              show_normals=True,
                              max_ray_len = ray_len
                              )

src_confocal = ConfocalRayFieldSource(origin = input_loc, 
                              direction= input_dir, 
                              number=50,
                              numerical_aperture=0.12,
                              display="wires",
                              opacity=0.1,
                              radius=10/2,
                              show_normals=True,
                              max_ray_len = ray_len,
                              working_dist= 0,
                              resolution = 10,#number of angle steps along the radius
                              angle=5
                              )


### Add probes ###

view_probe = RayCapturePlane(centre = input_loc,
                            direction = input_dir,
                           width = 20,
                           height= 20)

test_probe = RayCapturePlane(centre = tuple(np.array(input_loc) 
                                        + 10*normDir(input_dir)),
                            direction=input_dir,
                           width = 20,
                           height= 20)


### Build model ###
model = RayTraceModel(optics=[M1,M2,TM,view],
                        sources=[src_collimated,
                                #  src_confocal,
                                 ],
                        probes=[view_probe,test_probe])


# %%
###Now open the GUI###
model.configure_traits()

#%%
M1_trace1 = model.sources[0].traced_rays[0]
M2_trace1 = model.sources[0].traced_rays[1]
TM_trace = model.sources[0].traced_rays[2]
M2_trace2 = model.sources[0].traced_rays[3]
M1_trace2 = model.sources[0].traced_rays[4]
term_trace = model.sources[0].traced_rays[5]
input = model.sources[0].input_rays.origin
input_dir = model.sources[0].input_rays.direction


# %% Spot diagrams - z-y plane

plt.plot(input[:,2],input[:,1],'.')
plt.title("Input")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(M1_trace1.termination[:,2],M1_trace1.termination[:,1],'.')
plt.title("M1")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(M2_trace1.termination[:,2],M2_trace1.termination[:,1],'.')
plt.title("M2")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(TM_trace.termination[:,2],TM_trace.termination[:,1],'.')
plt.title("TM")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(M2_trace2.termination[:,2],M2_trace2.termination[:,1],'.')
plt.title("M2")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(M1_trace2.termination[:,2],M1_trace2.termination[:,1],'.')
plt.title("M1")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(term_trace.termination[:,2],term_trace.termination[:,1],'.')
plt.title("Ray termination")
plt.gca().set_aspect("equal")
plt.show()

# %% View ray interaction with optics and observation planes - x-y plane

plt.plot(M1_trace1.termination[:,0],M1_trace1.termination[:,1],'.')
plt.plot(M1_loc[0],M1_loc[1],'.')
plt.title("M1")
# plt.gca().set_aspect("equal")
plt.show()

plt.plot(M2_trace1.termination[:,0],M2_trace1.termination[:,1],'.')
plt.plot(M2_loc[0],M2_loc[1],'.')
plt.title("M2")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(TM_trace.termination[:,0],TM_trace.termination[:,1],'.')
plt.plot(TM_loc[0],TM_loc[1])
plt.title("TM")
plt.xlim(TM_loc[0]-2,TM_loc[0]+2)
plt.gca().set_aspect("equal")
plt.show()

plt.plot(M2_trace2.termination[:,0],M2_trace2.termination[:,1],'.')
plt.plot(M2_loc[0],M2_loc[1],'.')
plt.title("M2")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(M1_trace2.termination[:,0],M1_trace2.termination[:,1],'.')
plt.plot(M1_loc[0],M1_loc[1],'.')
plt.title("M1")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(term_trace.termination[:,0],term_trace.termination[:,1],'.')
plt.title("Termination")
mid_idx = int(np.floor(len(term_trace.termination[:,0])/2))
plt.xlim(term_trace.termination[mid_idx,0]-2,term_trace.termination[mid_idx,0]+2)
plt.gca().set_aspect("equal")
plt.show()


# %% Direction and position of rays at optics

plt.quiver(input[:,2],input[:,1],input_dir[:,2],input_dir[:,1])
plt.plot(input[:,2],input[:,1],'.')
plt.title("Input")
plt.gca().set_aspect("equal")
plt.show()

plt.quiver(M1_trace1.termination[:,2],M1_trace1.termination[:,1],M1_trace1.direction[:,2],M1_trace1.direction[:,1])
plt.plot(M1_trace1.termination[:,2],M1_trace1.termination[:,1],'.')
plt.title("M1")
plt.gca().set_aspect("equal")
plt.show()

plt.quiver(M2_trace1.termination[:,2],M2_trace1.termination[:,1],M2_trace1.direction[:,2],M2_trace1.direction[:,1])
plt.plot(M2_trace1.termination[:,2],M2_trace1.termination[:,1],'.')
plt.title("M2")
plt.gca().set_aspect("equal")
plt.show()

plt.quiver(TM_trace.termination[:,2],TM_trace.termination[:,1],TM_trace.direction[:,2],TM_trace.direction[:,1])
plt.plot(TM_trace.termination[:,2],TM_trace.termination[:,1],'.')
plt.title("TM")
plt.gca().set_aspect("equal")
plt.show()

plt.quiver(M2_trace2.termination[:,2],M2_trace2.termination[:,1],M2_trace2.direction[:,2],M2_trace2.direction[:,1])
plt.plot(M2_trace2.termination[:,2],M2_trace2.termination[:,1],'.')
plt.title("M2")
plt.gca().set_aspect("equal")
plt.show()

plt.quiver(M1_trace2.termination[:,2],M1_trace2.termination[:,1],M1_trace2.direction[:,2],M1_trace2.direction[:,1])
plt.plot(M1_trace2.termination[:,2],M1_trace2.termination[:,1],'.')
plt.title("M1")
plt.gca().set_aspect("equal")
plt.show()

plt.quiver(term_trace.termination[:,2],term_trace.termination[:,1],term_trace.direction[:,2],term_trace.direction[:,1])
plt.plot(term_trace.termination[:,2],term_trace.termination[:,1],'.')
plt.title("Termination")
plt.gca().set_aspect("equal")
plt.show()


# %%
###Now open the GUI###
model.configure_traits()

# %% Dictionary of attributes associated with probe ray bundle
dir(test_probe.captured[0].termination)


# %%
