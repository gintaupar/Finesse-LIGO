# Madison Simmonds, Camilla Compton, Hotter Stuff Workshop, Adelaide May 2024

# This uses https://github.com/danielbrown2/raypier_optics code for analysis,
# instructions for install in G2401138, under "Dan's Raypier Fork"

# Coordinates taken from D0900512
# Original Mode Matching done in T1000717
# Notes recorded in googledoc linked from G2401138 "ETM Transmon"

# TODO: Something strange happening if you specify a system that's too large, we're using from_mm = *1e-1 to fix this.
# TODO: Using spherical surfaces for M1 and M2, real systems are parabolic
# %%
import matplotlib.pyplot as plt
from raypier.api import RayTraceModel, GeneralLens, ParallelRaySource, SphericalFace, CircleShape, OpticalMaterial, HexagonalRayFieldSource, ConicFace, PlanarFace, PlanoConvexLens, RayCapturePlane,ConfocalRayFieldSource, ConfocalRaySource
from raypier.mirrors import PECMirror
from raypier.parabolics import OffAxisParabloid
import numpy as np
from matplotlib.ticker import EngFormatter

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

ETMY_loc=(200*from_mm,-80*from_mm,472.3*from_mm)
M1_loc=(199.959*from_mm,-79.99*from_mm,-827.451*from_mm)
FM1_loc=(200*from_mm,100.374*from_mm,-200.114*from_mm)
FM2_loc = (200*from_mm,110.761*from_mm,-858.48*from_mm)
M2_loc =(199.89*from_mm,282.467*from_mm,-260.469*from_mm)
LPM_loc =(200.018*from_mm,287.698*from_mm,-788.126*from_mm)
UPM_loc = (200*from_mm,492.686*from_mm,-775.049*from_mm)
TrM6_loc = (200*from_mm,498.686*from_mm,-698.849*from_mm)
TrM3_loc = (-127.66*from_mm,492.686*from_mm,-698.849*from_mm)
TrM8_loc = (-127.649*from_mm,492.686*from_mm,-521.049*from_mm)
VP_loc = (-1390.321*from_mm,72.395*from_mm,-660.403*from_mm)


# %% Distance between optics
etmy_to_M1 = np.sqrt(np.sum(np.square(np.array(M1_loc)-np.array(ETMY_loc))))
M1_to_FM1 = np.sqrt(np.sum(np.square(np.array(FM1_loc)-np.array(M1_loc))))
FM1_to_FM2 = np.sqrt(np.sum(np.square(np.array(FM2_loc)-np.array(FM1_loc))))
FM2_to_M2 = np.sqrt(np.sum(np.square(np.array(M2_loc)-np.array(FM2_loc))))
M2_to_LPM = np.sqrt(np.sum(np.square(np.array(LPM_loc)-np.array(M2_loc))))
LPM_to_UPM = np.sqrt(np.sum(np.square(np.array(UPM_loc)-np.array(LPM_loc))))
UPM_to_TrM6 = np.sqrt(np.sum(np.square(np.array(TrM6_loc)-np.array(UPM_loc))))
TrM6_to_TrM3 = np.sqrt(np.sum(np.square(np.array(TrM3_loc)-np.array(TrM6_loc))))
TrM3_to_TrM8 = np.sqrt(np.sum(np.square(np.array(TrM8_loc)-np.array(TrM3_loc))))
TrM8_to_VP = np.sqrt(np.sum(np.square(np.array(VP_loc)-np.array(TrM8_loc))))

print('etmy to M1',etmy_to_M1*1e1)
print('M1 to fm1',M1_to_FM1*1e1)
print('fm1 to fm2',FM1_to_FM2*1e1)
print('fm2 to m2',FM2_to_M2*1e1)
print('M2 to LPM',M2_to_LPM*1e1)
print('LPM to UPM',LPM_to_UPM*1e1)
print('UPM to trm6',UPM_to_TrM6*1e1)
print('trm6 to trm3',TrM6_to_TrM3*1e1)
print('trm3 to trm8',TrM3_to_TrM8*1e1)
print('trm8 to vp',TrM8_to_VP*1e1)

print('UPM to VP',(UPM_to_TrM6 + TrM6_to_TrM3 + TrM3_to_TrM8 + TrM8_to_VP)*1e1)
# %%
VP_to_airUPM_dist = 722*from_mm
airUPM_to_airLPM_dist = 419*from_mm
airLPM_to_L0_dist = 570*from_mm
L0_to_L1_dist = 610*from_mm
L1_to_L2_dist = 1079.0*from_mm
L2_to_HWS_dist = 2938.44652369105*from_mm #903.6*from_mm

airDirection = (0,0,1)

airUPM_loc = tuple(np.array(VP_loc) + VP_to_airUPM_dist*((np.array(VP_loc) - np.array(TrM8_loc))/np.linalg.norm(np.array(VP_loc) - np.array(TrM8_loc))))
airLPM_loc = tuple(np.array(airUPM_loc) + airUPM_to_airLPM_dist*np.array((0,-1,0)))
L0_loc = tuple(np.array(airLPM_loc) + airLPM_to_L0_dist*np.array(airDirection))
L1_loc = tuple(np.array(L0_loc) + L0_to_L1_dist*np.array(airDirection))
L2_loc = tuple(L1_loc + L1_to_L2_dist*np.array(airDirection))
HWS_loc = tuple(np.array(L2_loc) + L2_to_HWS_dist*np.array(airDirection))
Lin_loc = tuple(np.array(HWS_loc - 20*from_mm*np.array(airDirection)))

surface_loc = np.array([HWS_loc,L2_loc,L2_loc,L1_loc,L1_loc,L0_loc,L0_loc,airLPM_loc,airUPM_loc,VP_loc,TrM8_loc,TrM3_loc,TrM6_loc,UPM_loc,LPM_loc,M2_loc,FM2_loc,FM1_loc,M1_loc,ETMY_loc,ETMY_loc])

f_L0 = 2236.1*from_mm
f_L1 = 2236.1*from_mm
f_L2 = -382.520880995333*from_mm #-559*from_mm

m = OpticalMaterial(glass_name="N-BK7")




### Build a mirrors/lenses ###

print(mirror_direction_from_coords(UPM_loc, TrM6_loc, TrM3_loc))

# Lin_shape = CircleShape(radius=50.8*from_mm/2)
# Linf1 = SphericalFace(curvature = 1300*from_mm/2, z_height = 5.0*from_mm)
# Linf2 = PlanarFace()
# Linput = GeneralLens(name="L2",
#                  shape=Lin_shape,
#                 #  diameter=50.8*from_mm,
#                  surfaces=[Linf1,Linf2],
#                  centre=Lin_loc,
#                  direction=airDirection,
#                 #  curvature=f_L2/2,
#                  materials=[m])

L2_shape = CircleShape(radius=100*from_mm/2 #50.8*from_mm/2
                       )
L2f1 = SphericalFace(curvature = f_L2/2, z_height = 5.0*from_mm)
L2f2 = PlanarFace()
L2 = GeneralLens(name="L2",
                 shape=L2_shape,
                #  diameter=50.8*from_mm,
                 surfaces=[L2f1,L2f2],
                 centre=L2_loc,
                 direction=airDirection,
                #  curvature=f_L2/2,
                 materials=[m])

L1_shape = CircleShape(radius=100*from_mm/2 #50.8*from_mm/2
                       )
L1f1 = SphericalFace(curvature = f_L1/2, z_height = 5.0*from_mm)
L1f2 = PlanarFace()
L1 = GeneralLens(name="L1",
                 shape=L1_shape,
                # diameter=50.8*from_mm,
                 surfaces=[L1f1,L1f2],
                 centre=L1_loc,
                 direction=airDirection,
                #  curvature=f_L1/2,
                 materials=[m])

L0_shape = CircleShape(radius=100*from_mm/2 #50.8*from_mm/2
                       )
L0f1 = SphericalFace(curvature = f_L0/2, z_height = 5.0*from_mm)
L0f2 = PlanarFace()
L0 = GeneralLens(name="L0",
                 shape=L0_shape,
                #  diameter=50.8*from_mm,
                 surfaces=[L0f1,L0f2],
                 centre=L0_loc,
                 direction=airDirection,
                #  curvature=f_L0/2,
                 materials=[m])
print(mirror_direction_from_coords(airLPM_loc, L1_loc, L0_loc))

airLPM = PECMirror(name="airLPM",
                   centre=airLPM_loc,
                   direction = mirror_direction_from_coords(airUPM_loc,L0_loc,airLPM_loc),
                   diameter=150*from_mm, # 50.8*from_mm,
                   thickness=5.e-1)

airUPM = PECMirror(name="airUPM",
                   centre=airUPM_loc,
                   direction = mirror_direction_from_coords(VP_loc,airLPM_loc,airUPM_loc),
                   diameter=150*from_mm, #50.8*from_mm,
                   thickness=5.e-1)

vp_shape = CircleShape(radius=136*from_mm/2)
vpf = PlanarFace() #ConicFace(curvature = -4000*from_mm, conic_const=0)
VP = GeneralLens(name="VP",
                shape=vp_shape,
                surfaces=[vpf],
                centre=VP_loc,
                direction = tuple(np.array(VP_loc) - np.array(TrM8_loc)),
                )

TrM8 = PECMirror(name="TrM8",
                centre=TrM8_loc,
                direction=mirror_direction_from_coords(TrM3_loc, VP_loc, TrM8_loc),
                diameter=50.82*from_mm,
                thickness=5.e-1)

TrM3 = PECMirror(name="TrM3",
                centre=TrM3_loc,
                direction=mirror_direction_from_coords(TrM6_loc, TrM8_loc, TrM3_loc),
                diameter=50.82*from_mm,
                thickness=5.e-1)

TrM6 = PECMirror(name="TrM6",
                centre=TrM6_loc,
                direction=mirror_direction_from_coords(UPM_loc, TrM3_loc, TrM6_loc),
                diameter=50.82*from_mm,
                thickness=5.e-1)

UPM = PECMirror(name="UPM",
                centre=UPM_loc,
                direction=mirror_direction_from_coords(LPM_loc, TrM6_loc, UPM_loc),
                diameter=50.8*from_mm,
                thickness=5.e-1)

LPM = PECMirror(name="LPM",
                centre=LPM_loc,
                direction=mirror_direction_from_coords(M2_loc, UPM_loc, LPM_loc),
                diameter=50.8*from_mm,
                thickness=5.e-1)

m2_shape = CircleShape(radius=38.261*from_mm/2)
# m2f = ConicFace(curvature = 200e-1, conic_const=-1.0000000001, mirror=True)
m2f = SphericalFace(curvature = 200*from_mm, mirror=True)
M2 = GeneralLens(name="M2",
                 shape=m2_shape,
                 surfaces=[m2f],
                 centre=M2_loc,
                 direction=mirror_direction_from_coords(FM2_loc, LPM_loc, M2_loc))

FM2 = PECMirror(name="FM2",
                centre=FM2_loc,
                direction=mirror_direction_from_coords(FM1_loc, M2_loc, FM2_loc),
                diameter=88.392*from_mm,
                thickness=5.e-1)

FM1 = PECMirror(name="FM1",
                centre=FM1_loc,
                direction=mirror_direction_from_coords(M1_loc, FM2_loc, FM1_loc),
                diameter=152.4*from_mm,
                thickness=5.e-1)

m1_shape = CircleShape(radius=231.16*from_mm/2)
# m1f = ConicFace(curvature = -4000*from_mm, conic_const=-1.0000000001, mirror=True)
m1f = SphericalFace(curvature = -4000*from_mm, mirror=True)
M1 = GeneralLens(name="M1",
                 shape=m1_shape,
                 surfaces=[m1f],
                 centre=M1_loc,
                 direction=mirror_direction_from_coords(ETMY_loc, FM1_loc, M1_loc))

# ETMY as mirror
# ETMY = PECMirror(name="ETMY",
#                 centre=ETMY_loc,
#                 direction=(0,0,-1),
#                 diameter=250e-1,
#                 thickness=5.e-1)

# ETMY as transparent surface
# ETM_shape = CircleShape(radius=250*from_mm/2)
# ETMf = PlanarFace() 
# ETMY = GeneralLens(name="ETMY",
#                 shape=ETM_shape,
#                 surfaces=[ETMf],
#                 centre=ETMY_loc,
#                 direction = (0,0,-1),
#                 )

# ETM as glass substrate, with curved front surface
ETM_shape = CircleShape(radius=250*from_mm/2)
ETMf1 = PlanarFace() 
ETMf2 = SphericalFace(curvature = 2.2e6*from_mm, z_height = 200*from_mm)
ETMY = GeneralLens(name="ETMY",
                shape=ETM_shape,
                surfaces=[ETMf1,ETMf2],
                centre=ETMY_loc,
                direction = (0,0,1),
                materials=[m]
                )



### Add a source ###
src = HexagonalRayFieldSource(origin= HWS_loc, #(200e-1,-80e-1,450.3e-1), 
                              direction= tuple(-np.array(airDirection)), #(0,0,-1), # 
                              gauss_width=5.0,
                              display="wires",
                              opacity=0.1,
                              radius=2*from_mm,
                            # radius=20*from_mm,
                              resolution = 15,
                              show_normals=True)

src_parallelcross = ParallelRaySource(origin = HWS_loc, 
                        direction= tuple(-np.array(airDirection)), 
                        max_ray_len = 1000,
                        number = 8,
                        rings = 10,
                        radius = 6.48502190947465*from_mm, #2*from_mm,
                        display="wires",
                        opacity=0.3
                    )

src_confocalfield = ConfocalRayFieldSource(origin = HWS_loc, 
                              direction= tuple(-np.array(airDirection)), 
                              display="wires",
                              opacity=0.1,
                              show_normals=True,
                              max_ray_len = 1000,
                              working_dist= 0,
                              resolution = 10, # Total number of rays are (1 + resolution x 6)
                              angle=0.25 # half angle in degrees
                              )

src_confocalsource = ConfocalRaySource(focus = HWS_loc, 
                        direction= tuple(-np.array(airDirection)), 
                        max_ray_len = 1000,
                        number = 8, # number of angle steps along radius
                        rings = 10, # number of concentric rings of rays
                        display="wires",
                        opacity=0.3,
                        theta = 0.0014961424465528544 #0.25 #half angle in degrees
                    )

### Add probes ###

def normDir(init_ray_direction):
    norm_dir_ray = init_ray_direction/np.linalg.norm(init_ray_direction)
    return norm_dir_ray

input_probe = RayCapturePlane(centre = HWS_loc,
                            direction= airDirection,
                           width = 5*from_mm,
                           height= 5*from_mm)

ETM_probe = RayCapturePlane(centre = tuple(np.array(ETMY_loc) 
                                           + 0.00000000001*normDir(np.array([0,0,-1]))
                                           ),
                           direction = (0,0,-1),
                           width = 250e-1,
                           height=250e-1)


M1_probe = RayCapturePlane(centre = tuple(np.array(M1_loc) 
                                           + 0.001*normDir(np.array(mirror_direction_from_coords(ETMY_loc, FM1_loc, M1_loc)))
                                          ), # positive addition to centre location moves probe towards ETM
                            direction=tuple(np.array(mirror_direction_from_coords(ETMY_loc, FM1_loc, M1_loc))),
                           width = 231.16*from_mm,
                           height=231.16*from_mm)

M2_probe = RayCapturePlane(centre = tuple(np.array(M2_loc) 
                                        + 0.001*normDir(np.array(mirror_direction_from_coords(FM2_loc, LPM_loc, M2_loc)))
                                          ),
                            direction=mirror_direction_from_coords(FM2_loc, LPM_loc, M2_loc),
                           width = 38.261*from_mm,
                           height= 38.261*from_mm)


VP_probe = RayCapturePlane(centre = tuple(np.array(VP_loc) 
                                        + 0.000000001*normDir(np.array(tuple(np.array(VP_loc) - np.array(TrM8_loc))))
                                          ),
                            direction=tuple(np.array(VP_loc) - np.array(TrM8_loc)),
                           width = 136*from_mm,
                           height= 136*from_mm)

L0_probe = RayCapturePlane(centre = tuple(np.array(L0_loc) 
                                        + 0.000000001*normDir(np.array(airDirection))
                                          ),
                            direction= airDirection,
                           width = 50.8*from_mm,
                           height= 50.8*from_mm)

L1_probe = RayCapturePlane(centre = tuple(np.array(L1_loc) 
                                        + 0.000000001*normDir(np.array(airDirection))
                                          ),
                            direction= airDirection,
                           width = 50.8*from_mm,
                           height= 50.8*from_mm)

L2_probe = RayCapturePlane(centre = tuple(np.array(L2_loc) 
                                          + 0.000000001*normDir(np.array(airDirection))
                                          ),
                            direction= airDirection,
                           width = 50.8*from_mm,
                           height= 50.8*from_mm)

### Build model ###
ray_source = src_parallelcross
model = RayTraceModel(optics=[L2,L1,L0,airLPM,airUPM,VP,TrM8,TrM3,TrM6,UPM,LPM,M2,FM2,FM1,M1,ETMY
                              # ,Linput
                              ],
                        sources=[#src,
                                #  src_confocalsource,
                                #  src_confocalfield
                                src_parallelcross
                                 ],
                        probes=[ETM_probe, M1_probe,M2_probe,VP_probe,L0_probe,L1_probe,L2_probe,input_probe])

# %%
###Now open the GUI###

model.configure_traits()

# %%
surfaces = ["Input rays","L2 front surf","L2 back surf","L1 front surf","L1 back surf","L0 front surf","L0 back surf","Air LPM","Air UPM","View port","Transmon Mirror 8","Transmon Mirror 3","Transmon Mirror 6","UPM","LPM","M2","Folding mirror 2","Folding Mirror 1","M1","ETMY back surface","ETMY front surface","ray termination"
            ]
print(np.shape(surfaces))

# %% Analyse system

shape_rays = np.shape(ray_source.traced_rays[0].termination[:,0])

c_val = np.arange(0,shape_rays[0])
print(c_val)

# %%
fig, axs = plt.subplots(3,7 , figsize=[45,20],
                        constrained_layout=True
                        #gridspec_kw={'width_ratios':[1,1,1,1]}
                        )
fig.suptitle("Spot diagram in X-Y plane of rays at each surface within the system\n Single-pass trace (ETM front surface is termination plane, rays start at HWS)\n Rays colour-coded by position")
from_cm = 1e-2
formatter0 = EngFormatter(unit='m')
idx = 0


for i in np.arange(0,3):
    for j in np.arange(0,7):
      if i==0 and j==0:
        cf0 = axs[i,j].scatter((ray_source.input_rays.origin[:,0]-surface_loc[idx,0])*from_cm, (ray_source.input_rays.origin[:,1]-surface_loc[idx,1])*from_cm
                        , c=c_val, cmap="nipy_spectral"
                        )
        # cf0 = axs[i,j].scatter((input_probe.captured[0].termination[:,0]-surface_loc[idx,0])*from_cm, (input_probe.captured[0].termination[:,1]-surface_loc[idx,1])*from_cm
        #                 # , c=c_val, cmap="nipy_spectral"
        #                 )
        # axs[i,j].set_xlim(HWS_loc[0]-5e-9,HWS_loc[0]+5e-9)
        # axs[i,j].set_ylim(HWS_loc[1]-5e-9,HWS_loc[1]+5e-9)     
      else:
        cf0 = axs[i,j].scatter((ray_source.traced_rays[idx-1].termination[:,0]-surface_loc[idx,0])*from_cm, (ray_source.traced_rays[idx-1].termination[:,1]-surface_loc[idx,1])*from_cm
                          , c=c_val, cmap="nipy_spectral"
                          )
      axs[i,j].xaxis.set_major_formatter(formatter0)
      axs[i,j].yaxis.set_major_formatter(formatter0)
      # axs[i,j].set_aspect("equal")
      axs[i,j].set_xlabel("x ")
      axs[i,j].set_ylabel("y ")
      axs[i,j].grid()

      axs[i,j].set_title(
            f"@ {surfaces[idx]}")
      print(idx)
      idx = idx + 1
# %%
fig, axs = plt.subplots(1,2,figsize=[20,10],constrained_layout=True)
fig.suptitle("INput and output")
cf0 = axs[0].scatter((ray_source.input_rays.origin[:,0]-surface_loc[0,0])*from_cm, (ray_source.input_rays.origin[:,1]-surface_loc[0,1])*from_cm
                        , c=c_val, cmap="nipy_spectral"
                        )
axs[0].xaxis.set_major_formatter(formatter0)
axs[0].yaxis.set_major_formatter(formatter0)
axs[0].set_xlabel("x ")
axs[0].set_ylabel("y ")
axs[0].grid()
axs[0].set_title(f"@ HWS plane (input spots)")

cf0 = axs[1].scatter((ray_source.traced_rays[20].termination[:,0]
                      -surface_loc[20,0]
                      )*from_cm
                      , (ray_source.traced_rays[20].termination[:,1]
                         -surface_loc[20,1]
                         )*from_cm
                          , c=c_val, cmap="nipy_spectral"
                          )
axs[1].xaxis.set_major_formatter(formatter0)
axs[1].yaxis.set_major_formatter(formatter0)
axs[1].set_xlabel("x ")
axs[1].set_ylabel("y ")
axs[1].grid()
axs[1].set_title( f"@ ETMY front (curved) surface")


# %% Double pass model

# ETM_shape = CircleShape(radius=250*from_mm/2)
# ETMf = PlanarFace(mirror=True) 
# ETMY = GeneralLens(name="ETMY",
#                 shape=ETM_shape,
#                 surfaces=[ETMf],
#                 centre=ETMY_loc,
#                 direction = (0,0,-1),
#                 )

ETM_shape = CircleShape(radius=250*from_mm/2)
ETMf1 = PlanarFace() 
ETMf2 = PlanarFace(z_height = 200*from_mm,mirror=True) 
# ETMf2 = SphericalFace(curvature = 2.2e6*from_mm, z_height = 200*from_mm, mirror=True)
ETMY = GeneralLens(name="ETMY",
                shape=ETM_shape,
                surfaces=[ETMf1,ETMf2],
                centre=ETMY_loc,
                direction = (0,0,1),
                materials=[m]
                )

HWS_shape = CircleShape(radius=50.8*from_mm/2)
HWSf = PlanarFace()
HWS = GeneralLens(name="HWS",
                 shape=HWS_shape,
                 surfaces=[HWSf],
                 centre=HWS_loc,
                 direction=airDirection)

ray_source_double = src_parallelcross # set to match source included in model
model = RayTraceModel(optics=[HWS,L2,L1,L0,airLPM,airUPM,VP,TrM8,TrM3,TrM6,UPM,LPM,M2,FM2,FM1,M1,ETMY
                              # ,Linput
                              ],
                        sources=[#src,
                                #  src_confocalsource,
                                #  src_confocalfield
                                src_parallelcross
                                 ],
                        probes=[ETM_probe, M1_probe,M2_probe,VP_probe,L0_probe,L1_probe,L2_probe])


# %%
###Now open the GUI###

model.configure_traits()
 # %%
surfaces_double = ["HWS input","L2 front surf","L2 back surf","L1 front surf","L1 back surf","L0 front surf","L0 back surf","Air LPM","Air UPM","View port","Transmon Mirror 8","Transmon Mirror 3","Transmon Mirror 6","UPM","LPM","M2","Folding mirror 2","Folding Mirror 1","M1","ETMY back surface","ETMY front surface","ETMY back surface","M1","Folding Mirror 1","Folding Mirror 2","M2","LPM","UPM","Transmon Mirror 6","Transmon Mirror 3","Transmon Mirror 8","View port","Air UPM","Air LPM","L0 back surface","L0 front surface","L1 back surface","L1 front surface","L2 back surface","L2 front surface","HWS", "termination"]
print(np.shape(surfaces_double))

surface_loc_double = np.array([HWS_loc,L2_loc,L2_loc,L1_loc,L1_loc,L0_loc,L0_loc,airLPM_loc,airUPM_loc,VP_loc,TrM8_loc,TrM3_loc,TrM6_loc,UPM_loc,LPM_loc,M2_loc,FM2_loc,FM1_loc,M1_loc,ETMY_loc,ETMY_loc,ETMY_loc,M1_loc,FM1_loc,FM2_loc,M2_loc,LPM_loc,UPM_loc,TrM6_loc,TrM3_loc,TrM8_loc,VP_loc,airUPM_loc,airLPM_loc,L0_loc,L0_loc,L1_loc,L1_loc,L2_loc,L2_loc,HWS_loc])
print(np.shape(surface_loc_double))

# %%
np.shape(ray_source_double.traced_rays)#[38].termination[:,0]
# -surface_loc_double[38,0]
# %% Analyse system


shape_rays = np.shape(ray_source_double.traced_rays[0].termination[:,0])
c_val = np.arange(0,shape_rays[0])
# print(c_val)

fig, axs = plt.subplots(6,7 , figsize=[35,30],
                        constrained_layout=True
                        #gridspec_kw={'width_ratios':[1,1,1,1]}
                        )
fig.suptitle("Spot diagram in X-Y plane of rays at each surface within the system\n Double-pass trace (ETM front surface is reflection plane)\n ")
# fig.tight_layout()
from_cm = 1e-2
formatter0 = EngFormatter(unit='m')
idx = 0 

# for i in np.arange(0,13):
#   for j in np.arange(0,3):
#     cf0 = axs[i,j].scatter((ray_source_double.traced_rays[idx].termination[:,0]-surface_loc_double[idx,0])*from_cm, (ray_source_double.traced_rays[idx].termination[:,1]-surface_loc_double[idx,1])*from_cm
#                       # , c=c_val, cmap="nipy_spectral"
#                       )
#     axs[i,j].xaxis.set_major_formatter(formatter0)
#     axs[i,j].yaxis.set_major_formatter(formatter0)
#     # axs[i,j].set_aspect("equal")
#     axs[i,j].set_xlabel("x ")
#     axs[i,j].set_ylabel("y ")
#     axs[i,j].grid()

#     axs[i,j].set_title(
#         f"@ {surfaces_double[idx]}")
#     idx = idx + 1

for i in np.arange(0,6):
    for j in np.arange(0,7):
      if i==0 and j==0:
        cf0 = axs[i,j].scatter((ray_source_double.input_rays.origin[:,0]-surface_loc_double[idx,0])*from_cm, (ray_source_double.input_rays.origin[:,1]-surface_loc_double[idx,1])*from_cm
                        # , c=c_val, cmap="nipy_spectral"
                        )
        # cf0 = axs[i,j].scatter((input_probe.captured[0].termination[:,0]-surface_loc[idx,0])*from_cm, (input_probe.captured[0].termination[:,1]-surface_loc[idx,1])*from_cm
        #                 , c=c_val, cmap="nipy_spectral"
        #                 )
        # axs[i,j].set_xlim(HWS_loc[0]-5e-9,HWS_loc[0]+5e-9)
        # axs[i,j].set_ylim(HWS_loc[1]-5e-9,HWS_loc[1]+5e-9)
      else:
        cf0 = axs[i,j].scatter((ray_source_double.traced_rays[idx-1].termination[:,0]-surface_loc_double[idx,0])*from_cm, (ray_source_double.traced_rays[idx-1].termination[:,1]-surface_loc_double[idx,1])*from_cm
                          # , c=c_val, cmap="nipy_spectral"
                          )
      axs[i,j].xaxis.set_major_formatter(formatter0)
      axs[i,j].yaxis.set_major_formatter(formatter0)
      # axs[i,j].set_aspect("equal")
      axs[i,j].set_xlabel("x ")
      axs[i,j].set_ylabel("y ")
      axs[i,j].grid()

      axs[i,j].set_title(
          f"@ {surfaces_double[idx]}")
      idx = idx + 1

# %%

fig, axs = plt.subplots(1,3,figsize=[30,10],constrained_layout=True)
fig.suptitle("INput and output")
cf0 = axs[0].scatter((ray_source.input_rays.origin[:,0]-surface_loc_double[0,0])*from_cm, (ray_source.input_rays.origin[:,1]-surface_loc_double[0,1])*from_cm
                        , c=c_val, cmap="nipy_spectral"
                        )
axs[0].xaxis.set_major_formatter(formatter0)
axs[0].yaxis.set_major_formatter(formatter0)
axs[0].set_xlabel("x ")
axs[0].set_ylabel("y ")
axs[0].grid()
axs[0].set_title(f"@ HWS plane (input spots)")

cf0 = axs[1].scatter((ray_source.traced_rays[20].termination[:,0]
                      -surface_loc_double[20,0]
                      )*from_cm
                      , (ray_source.traced_rays[20].termination[:,1]
                         -surface_loc_double[20,1]
                         )*from_cm
                          , c=c_val, cmap="nipy_spectral"
                          )
axs[1].xaxis.set_major_formatter(formatter0)
axs[1].yaxis.set_major_formatter(formatter0)
axs[1].set_xlabel("x ")
axs[1].set_ylabel("y ")
axs[1].grid()
axs[1].set_title( f"@ ETMY front (curved) surface")

cf0 = axs[2].scatter((ray_source.traced_rays[40].termination[:,0]
                      -surface_loc_double[40,0]
                      )*from_cm
                      , (ray_source.traced_rays[40].termination[:,1]
                         -surface_loc_double[40,1]
                         )*from_cm
                          # , c=c_val, cmap="nipy_spectral"
                          )
axs[2].xaxis.set_major_formatter(formatter0)
axs[2].yaxis.set_major_formatter(formatter0)
axs[2].set_xlabel("x ")
axs[2].set_ylabel("y ")
axs[2].grid()
axs[2].set_title( f"@ HWS")


# %% Single pass from ETM

src_ETMorigin = ParallelRaySource(origin = tuple(np.array(ETMY_loc + 200*from_mm*np.array([0,0,1]))), 
                        direction= (0,0,-1), 
                        max_ray_len = 1000,
                        number = 8,
                        rings = 10,
                        radius = 5*from_mm, 
                        display="wires",
                        opacity=0.3
                    )

HWS_shape = CircleShape(radius=50.8*from_mm/2)
HWSf = PlanarFace()
HWS = GeneralLens(name="HWS",
                 shape=HWS_shape,
                 surfaces=[HWSf],
                 centre=HWS_loc,
                 direction=airDirection)

ray_source = src_ETMorigin # set to match source included in model
model = RayTraceModel(optics=[HWS,L2,L1,L0,airLPM,airUPM,VP,TrM8,TrM3,TrM6,UPM,LPM,M2,FM2,FM1,M1,ETMY
                              # ,Linput
                              ],
                        sources=[src_ETMorigin
                                 ],
                        probes=[ETM_probe, M1_probe,M2_probe,VP_probe,L0_probe,L1_probe,L2_probe])

# %%
###Now open the GUI###
model.configure_traits()

# %%
surfaces = ["Input rays","ETMY back surface","M1","FM1","FM2","M2","LPM","UPM","Transmon Mirror 6","Transmon Mirror 3","Transmon Mirror 8","View port","Air UPM","Air LPM","L0 back surface","L0 front surface","L1 back surface","L1 front surface","L2 back surface","L2 front surface","HWS","termination"]
print(np.shape(surfaces))

surface_loc = np.array([ETMY_loc,ETMY_loc,M1_loc,FM1_loc,FM2_loc,M2_loc,LPM_loc,UPM_loc,TrM6_loc,TrM3_loc,TrM8_loc,VP_loc,airUPM_loc,airLPM_loc,L0_loc,L0_loc,L1_loc,L1_loc,L2_loc,L2_loc,HWS_loc])
print(np.shape(surface_loc))

# %%
c_val = np.arange(0,8*10+1)
print(c_val)

fig, axs = plt.subplots(3,7 , figsize=[45,20],
                        constrained_layout=True
                        #gridspec_kw={'width_ratios':[1,1,1,1]}
                        )
fig.suptitle("Spot diagram in X-Y plane of rays at each surface within the system\n Single-pass trace (Rays start at ETM front surface, and terminate at the HWS plane)\n Rays colour-coded by position")
from_cm = 1e-2
formatter0 = EngFormatter(unit='m')
idx = 0


for i in np.arange(0,3):
    for j in np.arange(0,7):
      if i==0 and j==0:
        cf0 = axs[i,j].scatter((ray_source.input_rays.origin[:,0]-surface_loc[idx,0])*from_cm, (ray_source.input_rays.origin[:,1]-surface_loc[idx,1])*from_cm
                        , c=c_val, cmap="nipy_spectral"
                        )
        # cf0 = axs[i,j].scatter((input_probe.captured[0].termination[:,0]-surface_loc[idx,0])*from_cm, (input_probe.captured[0].termination[:,1]-surface_loc[idx,1])*from_cm
        #                 # , c=c_val, cmap="nipy_spectral"
        #                 )
        # axs[i,j].set_xlim(HWS_loc[0]-5e-9,HWS_loc[0]+5e-9)
        # axs[i,j].set_ylim(HWS_loc[1]-5e-9,HWS_loc[1]+5e-9)     
      else:
        cf0 = axs[i,j].scatter((ray_source.traced_rays[idx-1].termination[:,0]-surface_loc[idx,0])*from_cm, (ray_source.traced_rays[idx-1].termination[:,1]-surface_loc[idx,1])*from_cm
                          , c=c_val, cmap="nipy_spectral"
                          )
      axs[i,j].xaxis.set_major_formatter(formatter0)
      axs[i,j].yaxis.set_major_formatter(formatter0)
      # axs[i,j].set_aspect("equal")
      axs[i,j].set_xlabel("x ")
      axs[i,j].set_ylabel("y ")
      axs[i,j].grid()

      axs[i,j].set_title(
            f"@ {surfaces[idx]}")
      print(idx)
      idx = idx + 1
# %%
