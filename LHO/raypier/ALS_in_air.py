# Madison Simmonds, September 2024

# This uses https://github.com/danielbrown2/raypier_optics code for analysis,
# instructions for install in G2401138, under "Dan's Raypier Fork"

# Coordinates taken from T0900144
# Notes recorded in googledoc linked from T2400319 "LHO ETM HWS Beamtracing"


# %%
import matplotlib.pyplot as plt
from raypier.api import RayTraceModel, GeneralLens, ParallelRaySource, SphericalFace, CircleShape, OpticalMaterial, PlanarFace, PlanoConvexLens, RayCapturePlane, ConfocalRaySource, GaussianBeamRaySource
from raypier.mirrors import PECMirror
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

def normDir(init_ray_direction):
    norm_dir_ray = init_ray_direction/np.linalg.norm(init_ray_direction)
    return norm_dir_ray

from_mm=1e-1

m = OpticalMaterial(glass_name="N-BK7")

def gaussianbeamparameters(waist,wavelength):
    #enter beam waist in micron and wavelength in nm
    #function will return required values for beam waist, wavelength and rayleigh range to trace and observe Gaussian beam 
    beam_waist = 2*waist # micron
    radius = beam_waist/2000
    print('raypier radius',radius)
    wavelength = wavelength*1e-3 #micron
    zR = (np.pi*np.square(radius*1000)/wavelength)*1e-3 #mm
    theta0 = wavelength/(np.pi*radius*1000)
    print('raypier angle',theta0)
    return(beam_waist,wavelength,zR,theta0)

def spotsize(waist,rayleigh,z):
    #waist in micron, rayleigh and z in cm
    #function will return spot size in  micron
    spot_square = np.square(waist)*(1+np.square(z/rayleigh))
    spot = np.sqrt(spot_square)
    return(spot)

def biconvex532(f):
    n = 1.517551446918079
    return(2*f*(n-1))

lens_thickness = 5 #mm

# %% ALS in-air optics


VP_to_airUPM_dist = 722
VP_to_L7_dist = 3000
airUPM_to_airLPM_dist = 419
L7_to_L6_dist = 850
L6_to_L5_dist = 820
L5_to_waist_dist = 413.8


airDirection = (0,0,1)


input_loc = (0,0,0)
L5_loc = tuple(np.array(input_loc) + L5_to_waist_dist*np.array(airDirection) - np.array([0,0,lens_thickness/2]))
L6_loc = tuple(np.array(L5_loc) + L6_to_L5_dist*np.array(airDirection)- np.array([0,0,lens_thickness/2]))
L7_loc = tuple(np.array(L6_loc) + L7_to_L6_dist*np.array(airDirection)- np.array([0,0,lens_thickness/2]))
VP_loc = tuple(np.array(L7_loc) + VP_to_L7_dist*np.array(airDirection)- np.array([0,0,lens_thickness/2]))



plt.plot(L7_loc[2],L7_loc[1],'.',label = f"L7,({L7_loc})")
plt.plot(L6_loc[2],L6_loc[1],'.',label = f"L6,({L6_loc})")
plt.plot(L5_loc[2],L5_loc[1],'.',label = f" L5,({L5_loc})")
plt.plot(input_loc[2],input_loc[1],'.', label = f"input,({input_loc})")
plt.plot(VP_loc[2],VP_loc[1],'.',label = f"view port,({VP_loc})")
plt.quiver(input_loc[2],input_loc[1],airDirection[2],airDirection[1])
plt.legend()
plt.grid()
plt.show()







# %% Optics features
f_L7 = 1000
f_L6 = -200 
f_L5 = 350

R_L5 = biconvex532(f_L5)
R_L6 = biconvex532(f_L6)
R_L7 = biconvex532(f_L7)

print('RoC L5',R_L5)
print('RoC L6',R_L6)
print('RoC L7',R_L7)

# %% Beam features
waist = 213 #um
wavelength = 532 #nm

inputgauss = gaussianbeamparameters(waist,wavelength)

print(inputgauss)



# %% Probe locations

L6_to_waist = 366.2 #upstream
L7_to_waist = 1005.5 #upstream
w1_loc = tuple(np.array(L6_loc) - L6_to_waist*np.array(airDirection))
w2_loc = tuple(np.array(L7_loc) - L7_to_waist*np.array(airDirection))

plt.plot(L7_loc[2],L7_loc[1],'.',label = f"L7,({L7_loc})")
plt.plot(L6_loc[2],L6_loc[1],'.',label = f"L6,({L6_loc})")
plt.plot(L5_loc[2],L5_loc[1],'.',label = f" L5,({L5_loc})")
plt.plot(input_loc[2],input_loc[1],'.', label = f"input,({input_loc})")
plt.plot(VP_loc[2],VP_loc[1],'.',label = f"view port,({VP_loc})")
plt.quiver(input_loc[2],input_loc[1],airDirection[2],airDirection[1])
plt.plot(w1_loc[2],w1_loc[1],'.',label ='w1 - 366.2 upstream from L6')
plt.plot(w2_loc[2],w2_loc[1],'.',label=' w2 - 1005.5 upstreams from L7')
plt.legend()
plt.grid()
plt.show()

# input to L5 
L5_spacing = L5_to_waist_dist/10

# L5 to L6
L6_spacing = L6_to_L5_dist/10

# L6 to L7 
L7_spacing = L7_to_L6_dist/10

# L7 to VP

VP_spacing = VP_to_L7_dist/10

# %%
### Build ALS in-air system ###

#### Optics #####
L7_shape = CircleShape(radius=50.8/2)
L7f1 = SphericalFace(curvature = -R_L7)
L7f2 = SphericalFace(curvature = R_L7,z_height = lens_thickness)
L7 = GeneralLens(name="L7",
                 shape=L7_shape,
                 surfaces=[L7f1,L7f2],
                 centre=L7_loc,
                 direction=airDirection,
                 materials=[m])

L6_shape = CircleShape(radius=50.8/2
                       )
L6f1 = SphericalFace(curvature = -R_L6)
L6f2 = SphericalFace(curvature = R_L6, z_height = lens_thickness)
L6 = GeneralLens(name="L6",
                 shape=L6_shape,
                 surfaces=[L6f1,L6f2],
                 centre=L6_loc,
                 direction=airDirection,
                 materials=[m])

L5_shape = CircleShape(radius=50.8/2
                       )
L5f1 = SphericalFace(curvature = -R_L5)
L5f2 = SphericalFace(curvature = R_L5, z_height = lens_thickness)
L5 = GeneralLens(name="L5",
                 shape=L5_shape,
                 surfaces=[L5f1,L5f2],
                 centre=L5_loc,
                 direction=airDirection,
                 materials=[m])

##### Beam Source #####

src_gaussian_beam = GaussianBeamRaySource(
                                            origin=input_loc,
                                            direction = airDirection,
                                            beam_waist = inputgauss[0],
                                            wavelength = inputgauss[1],
                                            max_ray_len = 30000,
                                            working_distance = 0
                                            )


src_parallelcrossALS = ParallelRaySource(origin = input_loc, 
                        direction= tuple(np.array(airDirection)), 
                        max_ray_len = 1000,
                        number = 4,
                        rings = 10,
                        radius = waist, 
                        display="wires",
                        opacity=0.3,
                    )

src_confocalcross = ConfocalRaySource(origin = input_loc, 
                        direction= airDirection, 
                        max_ray_len = 1000,
                        number = 4,
                        rings = 10,
                        radius = inputgauss[0]/2,
                        display="wires",
                        opacity=0.3,
                        theta = np.rad2deg(inputgauss[3]),
                        wavelength = inputgauss[1] #um
                    )                    

# Add probes to check waist locations # 

w1_probe = RayCapturePlane(centre = w1_loc,
                            direction=tuple(airDirection),
                           width = 50.8,
                           height=50.8)
w2_probe = RayCapturePlane(centre = w2_loc,
                            direction=tuple(airDirection),
                           width = 50.8,
                           height=50.8)

probe_width = 50.8

# Lens probes
L5_probe = RayCapturePlane(centre = L5_loc,
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L6_probe = RayCapturePlane(centre = L6_loc,
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L7_probe = RayCapturePlane(centre = L7_loc,
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

VP_probe = RayCapturePlane(centre = VP_loc,
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

# Input to L5 probes
L5_range_probe1 = RayCapturePlane(centre = tuple(np.array(input_loc) + 1*L5_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L5_range_probe2 = RayCapturePlane(centre = tuple(np.array(input_loc) + 2*L5_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)                          

L5_range_probe3 = RayCapturePlane(centre = tuple(np.array(input_loc) + 3*L5_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L5_range_probe4 = RayCapturePlane(centre = tuple(np.array(input_loc) + 4*L5_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L5_range_probe5 = RayCapturePlane(centre = tuple(np.array(input_loc) + 5*L5_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L5_range_probe6 = RayCapturePlane(centre = tuple(np.array(input_loc) + 6*L5_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L5_range_probe7 = RayCapturePlane(centre = tuple(np.array(input_loc) + 7*L5_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L5_range_probe8 = RayCapturePlane(centre = tuple(np.array(input_loc) + 8*L5_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L5_range_probe9 = RayCapturePlane(centre = tuple(np.array(input_loc) + 9*L5_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L5_range_probe10 = RayCapturePlane(centre = tuple(np.array(input_loc) + 10*L5_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)


# L5 to L6 probes
L6_range_probe1 = RayCapturePlane(centre = tuple(np.array(L5_loc) + 1*L6_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L6_range_probe2 = RayCapturePlane(centre = tuple(np.array(L5_loc) + 2*L6_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L6_range_probe3 = RayCapturePlane(centre = tuple(np.array(L5_loc) + 3*L6_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L6_range_probe4 = RayCapturePlane(centre = tuple(np.array(L5_loc) + 4*L6_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L6_range_probe5 = RayCapturePlane(centre = tuple(np.array(L5_loc) + 5*L6_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L6_range_probe6 = RayCapturePlane(centre = tuple(np.array(L5_loc) + 6*L6_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L6_range_probe7 = RayCapturePlane(centre = tuple(np.array(L5_loc) + 7*L6_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L6_range_probe8 = RayCapturePlane(centre = tuple(np.array(L5_loc) + 8*L6_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L6_range_probe9 = RayCapturePlane(centre = tuple(np.array(L5_loc) + 9*L6_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L6_range_probe10 = RayCapturePlane(centre = tuple(np.array(L5_loc) + 10*L6_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

# L6 to L7 
L7_range_probe1 = RayCapturePlane(centre = tuple(np.array(L6_loc) + 1*L7_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L7_range_probe2 = RayCapturePlane(centre = tuple(np.array(L6_loc) + 2*L7_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L7_range_probe3 = RayCapturePlane(centre = tuple(np.array(L6_loc) + 3*L7_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L7_range_probe4 = RayCapturePlane(centre = tuple(np.array(L6_loc) + 4*L7_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L7_range_probe5 = RayCapturePlane(centre = tuple(np.array(L6_loc) + 5*L7_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L7_range_probe6 = RayCapturePlane(centre = tuple(np.array(L6_loc) + 6*L7_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L7_range_probe7 = RayCapturePlane(centre = tuple(np.array(L6_loc) + 7*L7_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L7_range_probe8 = RayCapturePlane(centre = tuple(np.array(L6_loc) + 8*L7_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L7_range_probe9 = RayCapturePlane(centre = tuple(np.array(L6_loc) + 9*L7_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

L7_range_probe10 = RayCapturePlane(centre = tuple(np.array(L6_loc) + 10*L7_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

# L7 to VP range
VP_range_probe1 = RayCapturePlane(centre = tuple(np.array(L7_loc) + 1*VP_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

VP_range_probe2 = RayCapturePlane(centre = tuple(np.array(L7_loc) + 2*VP_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

VP_range_probe3 = RayCapturePlane(centre = tuple(np.array(L7_loc) + 3*VP_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

VP_range_probe4 = RayCapturePlane(centre = tuple(np.array(L7_loc) + 4*VP_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

VP_range_probe5 = RayCapturePlane(centre = tuple(np.array(L7_loc) + 5*VP_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

VP_range_probe6 = RayCapturePlane(centre = tuple(np.array(L7_loc) + 6*VP_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

VP_range_probe7 = RayCapturePlane(centre = tuple(np.array(L7_loc) + 7*VP_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

VP_range_probe8 = RayCapturePlane(centre = tuple(np.array(L7_loc) + 8*VP_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

VP_range_probe9 = RayCapturePlane(centre = tuple(np.array(L7_loc) + 9*VP_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)

VP_range_probe10 = RayCapturePlane(centre = tuple(np.array(L7_loc) + 10*VP_spacing*np.array(airDirection)),
                           direction = airDirection,
                           width = probe_width,
                           height=probe_width)



### Build model ###
ray_source = src_parallelcrossALS
model = RayTraceModel(optics=[L5,L6,L7,
                              ],
                        sources=[
                                # src_parallelcrossALS,
                                src_gaussian_beam,
                                # src_confocalcross
                                 ],
                        probes=[w1_probe,w2_probe,
                        L5_probe,L6_probe,L7_probe,VP_probe,
                        L5_range_probe1,L5_range_probe2,L5_range_probe3,L5_range_probe4,L5_range_probe5,L5_range_probe6,L5_range_probe7,L5_range_probe8,L5_range_probe9,L5_range_probe10,
                        L6_range_probe1,L6_range_probe2,L6_range_probe3,L6_range_probe4,L6_range_probe5,L6_range_probe6,L6_range_probe7,L6_range_probe8,L6_range_probe9,L6_range_probe10,
                        L7_range_probe1,L7_range_probe2,L7_range_probe3,L7_range_probe4,L7_range_probe5,L7_range_probe6,L7_range_probe7,L7_range_probe8,L7_range_probe9,L7_range_probe10,
                        VP_range_probe1,VP_range_probe2,VP_range_probe3,VP_range_probe4,VP_range_probe5,VP_range_probe6,VP_range_probe7,VP_range_probe8,VP_range_probe9,VP_range_probe10
                        ])

# %%
###Now open the GUI###

model.configure_traits()


# %%
formatter0 = EngFormatter(unit='m')
mu_unicode = "\u03BC"

fig, evolve = plt.subplots(1,1,figsize = [30,12])
fig.suptitle(f'ALS in-air optics chain\n f5 = {f_L5}mm, f6 = {f_L6}mm, f7 = {f_L7}mm\n Beam input waist = {waist}{mu_unicode}m, wavelength = {wavelength}nm,\n  ',fontsize=20)


evolve.grid()
evolve.tick_params(labelsize=15)


### Full range view planes

evolve.scatter(L5_range_probe1.captured[0].termination[:,2]*1e-3,L5_range_probe1.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L5_range_probe2.captured[0].termination[:,2]*1e-3,L5_range_probe2.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L5_range_probe3.captured[0].termination[:,2]*1e-3,L5_range_probe3.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L5_range_probe4.captured[0].termination[:,2]*1e-3,L5_range_probe4.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L5_range_probe5.captured[0].termination[:,2]*1e-3,L5_range_probe5.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L5_range_probe6.captured[0].termination[:,2]*1e-3,L5_range_probe6.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L5_range_probe7.captured[0].termination[:,2]*1e-3,L5_range_probe7.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L5_range_probe8.captured[0].termination[:,2]*1e-3,L5_range_probe8.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L5_range_probe9.captured[0].termination[:,2]*1e-3,L5_range_probe9.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L5_range_probe10.captured[0].termination[:,2]*1e-3,L5_range_probe10.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

# L6 view planes

evolve.scatter(L6_range_probe1.captured[0].termination[:,2]*1e-3,L6_range_probe1.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L6_range_probe2.captured[0].termination[:,2]*1e-3,L6_range_probe2.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L6_range_probe3.captured[0].termination[:,2]*1e-3,L6_range_probe3.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L6_range_probe4.captured[0].termination[:,2]*1e-3,L6_range_probe4.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L6_range_probe5.captured[0].termination[:,2]*1e-3,L6_range_probe5.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L6_range_probe6.captured[0].termination[:,2]*1e-3,L6_range_probe6.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L6_range_probe7.captured[0].termination[:,2]*1e-3,L6_range_probe7.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L6_range_probe8.captured[0].termination[:,2]*1e-3,L6_range_probe8.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L6_range_probe9.captured[0].termination[:,2]*1e-3,L6_range_probe9.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L6_range_probe10.captured[0].termination[:,2]*1e-3,L6_range_probe10.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

# L7 view planes

evolve.scatter(L7_range_probe1.captured[0].termination[:,2]*1e-3,L7_range_probe1.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L7_range_probe2.captured[0].termination[:,2]*1e-3,L7_range_probe2.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L7_range_probe3.captured[0].termination[:,2]*1e-3,L7_range_probe3.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L7_range_probe4.captured[0].termination[:,2]*1e-3,L7_range_probe4.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L7_range_probe5.captured[0].termination[:,2]*1e-3,L7_range_probe5.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L7_range_probe6.captured[0].termination[:,2]*1e-3,L7_range_probe6.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L7_range_probe7.captured[0].termination[:,2]*1e-3,L7_range_probe7.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L7_range_probe8.captured[0].termination[:,2]*1e-3,L7_range_probe8.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L7_range_probe9.captured[0].termination[:,2]*1e-3,L7_range_probe9.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(L7_range_probe10.captured[0].termination[:,2]*1e-3,L7_range_probe10.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

# VP

evolve.scatter(VP_range_probe1.captured[0].termination[:,2]*1e-3,VP_range_probe1.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(VP_range_probe2.captured[0].termination[:,2]*1e-3,VP_range_probe2.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(VP_range_probe3.captured[0].termination[:,2]*1e-3,VP_range_probe3.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(VP_range_probe4.captured[0].termination[:,2]*1e-3,VP_range_probe4.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(VP_range_probe5.captured[0].termination[:,2]*1e-3,VP_range_probe5.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(VP_range_probe6.captured[0].termination[:,2]*1e-3,VP_range_probe6.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(VP_range_probe7.captured[0].termination[:,2]*1e-3,VP_range_probe7.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(VP_range_probe8.captured[0].termination[:,2]*1e-3,VP_range_probe8.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(VP_range_probe9.captured[0].termination[:,2]*1e-3,VP_range_probe9.captured[0].termination[:,1]*1e-3,c='#8ac0fc')

evolve.scatter(VP_range_probe10.captured[0].termination[:,2]*1e-3,VP_range_probe10.captured[0].termination[:,1]*1e-3,c='#8ac0fc')


### Special planes

evolve.plot(np.array([L5_range_probe1.captured[0].origin[0,2]*1e-3,L5_range_probe1.captured[0].origin[0,2]*1e-3]),
np.array([min(L5_range_probe1.captured[0].origin[:,1]*1e-3),max(L5_range_probe1.captured[0].origin[:,1]*1e-3)]),
label = f'input',linewidth=5)

evolve.plot(np.array([L5_loc[2]*1e-3,L5_loc[2]*1e-3]),
np.array([min(L5_probe.captured[0].termination[:,1]*1e-3),max(L5_probe.captured[0].termination[:,1]*1e-3)]),
c='m',label=f"L5",linewidth=5)


evolve.plot(np.array([L6_loc[2]*1e-3,L6_loc[2]*1e-3]),
np.array([min(L6_range_probe10.captured[0].termination[:,1]*1e-3),max(L6_range_probe10.captured[0].termination[:,1]*1e-3)]),
c='#fe0261',label='L6',linewidth=5)

evolve.plot(np.array([L7_loc[2]*1e-3,L7_loc[2]*1e-3]),np.array([min(L7_probe.captured[0].termination[:,1]*1e-3),max(L7_probe.captured[0].termination[:,1]*1e-3)]),c="#7c027e",label='L7',linewidth=5)

evolve.plot(np.array([VP_loc[2]*1e-3,VP_loc[2]*1e-3]),np.array([min(VP_probe.captured[0].termination[:,1]*1e-3),max(VP_probe.captured[0].termination[:,1]*1e-3)]),c="#c6058e",label='VP',linewidth=5)

# Claimed waist locations

evolve.plot(np.array([w1_loc[2]*1e-3,w1_loc[2]*1e-3]),np.array([min(w1_probe.captured[0].termination[:,1]*1e-3),max(w1_probe.captured[0].termination[:,1]*1e-3)]),c="#f6170c",label=f'Waist 1; claimed size = 271{mu_unicode}m\n Actual size = {max(w1_probe.captured[0].termination[:,1])*1e3:.4}{mu_unicode}m',linewidth=3)

evolve.plot(np.array([w2_loc[2]*1e-3,w2_loc[2]*1e-3]),np.array([min(w2_probe.captured[0].termination[:,1]*1e-3),max(w2_probe.captured[0].termination[:,1]*1e-3)]),c="#f68c0c",label=f'Waist 2; claimed size = 76{mu_unicode}m\n Actual size = {max(w2_probe.captured[0].termination[:,1])*1e3:.4}',linewidth=3)

evolve.xaxis.set_major_formatter(formatter0)
evolve.yaxis.set_major_formatter(formatter0)

evolve.legend(loc='upper left',fontsize=17)

evolve.set_xlabel('z ',fontsize=20)
evolve.set_ylabel('y ',fontsize=20)


# %%




# %%
