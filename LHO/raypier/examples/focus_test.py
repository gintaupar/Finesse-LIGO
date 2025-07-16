# Madison Simmonds, Camilla Compton, Hotter Stuff Workshop, Adelaide May 2024

# This uses https://github.com/danielbrown2/raypier_optics code for analysis,
# instructions for install in G2401138, under "Dan's Raypier Fork"

# Script demonstrating raypier functionality for single-optic system with confocal input beam

# Input beam located at the focal point of the mirror
# Transparent 'view' optic placed at focal plane of mirror
# With the input beam starting at a focus at the focal point of the mirror, the rays will return to the focal plane collimated

# %%
import matplotlib.pyplot as plt
from raypier.api import RayTraceModel, GeneralLens, ParallelRaySource, SphericalFace, CircleShape, OpticalMaterial, HexagonalRayFieldSource, ConicFace, PlanarFace, PlanoConvexLens, RayCapturePlane, DistortionFace,ConfocalRayFieldSource,ConfocalRaySource
from raypier.faces import BaseFace
from raypier.distortions import ZernikeSeries,SimpleTestZernikeJ7
from raypier.mirrors import PECMirror
from raypier.parabolics import OffAxisParabloid
import numpy as np


# %%
# Useful functions
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

### Build mirrors/lenses ###
m1_shape = CircleShape(radius=60/2)
m1f = ConicFace(curvature = -2*f1, conic_const=-1.0000000001, mirror=True)
M1 = GeneralLens(name="M1",
                 shape=m1_shape,
                 surfaces=[m1f],
                 centre=M1_loc,
                 direction=M1_dir)


# View plane as transparent surface
view_shape = CircleShape(radius=60/2)
viewf = PlanarFace(mirror=False) 
view = GeneralLens(name="view",
                shape=view_shape,
                surfaces=[viewf],
                centre=input_loc,
                direction = input_target_dir,
                )

### Add a source ###

# field source produces rays in hex pattern
src = ConfocalRayFieldSource(origin = input_loc, 
                              direction= input_target_dir, 
                              display="wires",
                              opacity=0.1,
                              show_normals=True,
                              max_ray_len = 2*f1,
                              working_dist= 0, # how far from origin the rays are focused
                              resolution = 3, # Total number of rays are 1 + resolution x 6
                              angle=5 # half angle in deg
                              )

# ray source produces rays in spoke pattern
src_raysource = ConfocalRaySource(focus = input_loc, 
                        direction= input_target_dir, 
                        max_ray_len = 1000,
                        number = 10, # number of angle steps along radius
                        rings = 10, # number of concentric rings of rays
                        display="wires",
                        opacity=0.3,
                        theta = 5 #half angle in degrees
                    )


### Add probes ###
# Adding in view plane to observe rays at arbitary location within system
view_probe = RayCapturePlane(centre = tuple(np.array(input_loc) 
                                           + 10*normDir(input_target_dir)
                                           ),
                           direction = input_target_dir,
                           width = 30,
                           height=30)


### Build model ###
model = RayTraceModel(optics=[
                            M1,
                              view
                              ],
                        sources=[#src,
                                 src_raysource
                                 ],
                        probes=[view_probe])


# %%
###Now open the GUI###
model.configure_traits()



# %%
M1_trace = src.traced_rays[0]
view_trace = src.traced_rays[1]
termination_trace = src.traced_rays[2]

# %% Spot diagrams - z-y plane

plt.plot(M1_trace.termination[:,2],M1_trace.termination[:,1],'.')
plt.title("@ Mirror")
plt.xlabel("z [cm]")
plt.ylabel("y [cm]")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(view_trace.termination[:,2],view_trace.termination[:,1],'.')
plt.title("@ View plane")
plt.xlabel("z [cm]")
plt.ylabel("y [cm]")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(termination_trace.termination[:,2],termination_trace.termination[:,1],'.')
plt.title("@ Ray termination")
plt.xlabel("z [cm]")
plt.ylabel("y [cm]")
plt.gca().set_aspect("equal")
plt.show()


# %% Optic location - x-y plane

plt.plot(M1_trace.termination[:,0],M1_trace.termination[:,1],'.')
plt.plot(M1_loc[0],M1_loc[1],'.')
plt.title("Mirror location")
plt.xlim(M1_loc[0] - 10, M1_loc[0]+10)
plt.xlabel("x [cm]")
plt.ylabel("y [cm]")
plt.gca().set_aspect("equal")
plt.show()

plt.plot(view_trace.termination[:,0],view_trace.termination[:,1] ,'.')
plt.plot(input_loc[0],input_loc[1],'.')
plt.title("View plane location")
plt.xlabel("x [cm]")
plt.ylabel("y [cm]")
plt.xlim(input_loc[0] - 10, input_loc[0]+10)
plt.gca().set_aspect("equal")
plt.show()

plt.plot(termination_trace.termination[:,0],termination_trace.termination[:,1],'.')
plt.title("Ray termination location") # Note: rays terminate at 'max_ray_len' distance from last optic
plt.xlabel("x [cm]")
plt.ylabel("y [cm]")
plt.gca().set_aspect("equal")
plt.show()




# %% Direction and position of rays at optics

plt.quiver(M1_trace.termination[:,2],M1_trace.termination[:,1],M1_trace.direction[:,2],M1_trace.direction[:,1])
plt.plot(M1_trace.termination[:,2],M1_trace.termination[:,1],'.')
plt.title("Direction of rays @ Mirror")
plt.xlabel("z [cm]")
plt.ylabel("y [cm]")
plt.gca().set_aspect("equal")
plt.show()


plt.quiver(view_trace.termination[:,2],view_trace.termination[:,1],view_trace.direction[:,2],view_trace.direction[:,1])
plt.plot(view_trace.termination[:,2],view_trace.termination[:,1],'.')
plt.title("Direction of rays @ View plane")
plt.xlabel("z [cm]")
plt.ylabel("y [cm]")
plt.gca().set_aspect("equal")
plt.show()

plt.quiver(termination_trace.termination[:,2],termination_trace.termination[:,1],termination_trace.direction[:,2],termination_trace.direction[:,1])
plt.plot(termination_trace.termination[:,2],termination_trace.termination[:,1],'.')
plt.title("Direction of rays @ Ray termination plane")
plt.xlabel("z [cm]")
plt.ylabel("y [cm]")
plt.gca().set_aspect("equal")
plt.show()


# %% View optical path length of rays between two surfaces
fig, axs = plt.subplots()
cf0 = axs.scatter(M1_trace.termination[:,2], M1_trace.termination[:,1]
                  , c=M1_trace.length - f1
                  )

axs.set_aspect("equal")
cb = fig.colorbar(cf0, ax=axs)
axs.set_xlabel("z [cm]")
axs.set_ylabel("y [cm]")
cb.ax.set_ylabel(r"OPD - f1 [cm]")
axs.set_title(
    f"@ M1")



fig, axs = plt.subplots()
cf0 = axs.scatter(view_trace.termination[:,2], view_trace.termination[:,1]
                  , c=view_trace.length - f1
                  )
axs.set_aspect("equal")
cb = fig.colorbar(cf0, ax=axs)
axs.set_xlabel("z [cm]")
axs.set_ylabel("y [cm]")
cb.ax.set_ylabel(r"OPD - f1 [cm]")
axs.set_title(
    f"@ View plane")


# %%

print(M1_trace.length[181])
print(view_trace.length[181])
print(M1_trace.length[180])
print(view_trace.length[180])
print(M1_trace.termination[181,:])
print(view_trace.termination[181,:])

total_OPD = (M1_trace.length + view_trace.length) -2*f1


fig, axs = plt.subplots()
cf0 = axs.scatter(#src.input_rays.origin[:,2],src.input_rays.origin[:,1]
    view_trace.termination[:,2], view_trace.termination[:,1]
                  , c=total_OPD 
                  )
axs.set_aspect("equal")
cb = fig.colorbar(cf0, ax=axs)
axs.set_xlabel("z [cm]")
axs.set_ylabel("y [cm]")
# axs.set_ylim(view_trace.termination[20,1]-5,view_trace.termination[20,1]+5)
cb.ax.set_ylabel(r"OPD - 600cm [cm]")
axs.set_title(
    f"Cumulative OPD")



# %%
###Now open the GUI###
model.configure_traits()

# %%
