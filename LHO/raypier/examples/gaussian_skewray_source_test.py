# Madison Simmonds, September 2024

# This uses https://github.com/danielbrown2/raypier_optics code for analysis,
# instructions for install in G2401138, under "Dan's Raypier Fork"

# Script demonstrating raypier beam source class "GaussianBeamRaySource"

# This class traces a set of skew rays through a system that describe the envelope of a Gaussian beam
# This script provides a function that is able to calculate the input parameters required for the trace to return results in units of cm, given a beam waist in micron and a wavelength in nm

# The results of the ray trace are then plotted to observe the spots at relevant locations


# %%
import matplotlib.pyplot as plt
from raypier.api import RayTraceModel,RayCapturePlane,GaussianBeamRaySource
import numpy as np
from matplotlib.ticker import EngFormatter


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


# %%
def gaussianbeamparameters(waist,wavelength):
    #enter beam waist in micron and wavelength in nm
    #function will return required values for beam waist, wavelength and rayleigh range to trace and observe Gaussian beam 
    beam_waist = 2*waist*1e-1 # deci-micron
    radius = beam_waist/2000
    print('raypier radius',radius)
    wavelength = wavelength*1e-4 #deci-micron
    zR = (np.pi*np.square(radius*1000)/wavelength)*1e-3 #cm
    theta0 = wavelength/(np.pi*radius*1000)
    print('raypier angle',theta0)
    return(beam_waist,wavelength,zR,theta0)

def spotsize(waist,rayleigh,z):
    #waist in micron, rayleigh and z in cm
    #function will return spot size in  micron
    spot_square = np.square(waist)*(1+np.square(z/rayleigh))
    spot = np.sqrt(spot_square)
    return(spot)


# %%
input_waist = 300 #micron
input_wavelength = 532 #nm


# %%
# true value calc

true_angle = input_wavelength*1e-9/(np.pi*input_waist*1e-6)
true_range = np.pi*np.square(input_waist*1e-6)/(input_wavelength*1e-9)
true_spot = np.sqrt(np.square(input_waist*1e-6)*(1+np.square(true_range/true_range)))

print("true input waist [m]",300*1e-6)
print("true angle",true_angle)
print("true rayleigh range",true_range,"m")
print("true spot size at rayleigh range",true_spot,"m")



gausstest = gaussianbeamparameters(input_waist,input_wavelength)
print(gausstest)
print("Raypier input beam waist",gausstest[0])
print("Raypier input wavelength",gausstest[1])
print("Raypier output rayleigh range",gausstest[2],"cm")
print("Raypier output far field diffraction angle",gausstest[3])



half_range = spotsize(input_waist,gausstest[2],0.5*gausstest[2])
full_range = spotsize(input_waist,gausstest[2],gausstest[2])
past_range = spotsize(input_waist,gausstest[2],1.5*gausstest[2])
print("Spotsize at half rayleigh range",half_range,"micron")
print("Spot size at the rayleigh range",full_range,"micron")
print("Spot size at 1.5x rayleigh range",past_range,"micron")

# %% Collimated input beam


### Add a source ###
src_gaussian_beam = GaussianBeamRaySource(
                                            origin=(0,0,0),
                                            direction = (0,0,1),
                                            beam_waist = gausstest[0],
                                            wavelength = gausstest[1],
                                            max_ray_len = 30000,
                                            working_distance = 0

                                        )


### Add probes ###
view_probe1 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 0.1*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)

view_probe2 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 0.2*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)

view_probe3 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 0.3*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)

view_probe4 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 0.4*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)

view_probe5 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 0.5*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)

view_probe6 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 0.6*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)


view_probe7 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 0.7*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)


view_probe8 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 0.8*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)


view_probe9 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 0.9*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)


view_probe10 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 1.0*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)


view_probe11 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 1.5*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)                                                 

view_probe12 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 5*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)                                                            
view_probe13 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 10*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500) 

view_probe14 = RayCapturePlane(centre = tuple(np.array([0,0,0]) + 15*gausstest[2]*np.array([0,0,1])),
                           direction = (0,0,1),
                           width = 2*500,
                           height=2*500)                            


### Build model ###
model_parallelcross = RayTraceModel(optics=[],
                        sources=[src_gaussian_beam]
                                 ,probes=[view_probe1,view_probe2,view_probe3,view_probe4,view_probe5,view_probe6,view_probe7,view_probe8,view_probe9,view_probe10,view_probe11,view_probe12,view_probe13,view_probe14],
                                 results=[]
                        )


# %%
###Now open the GUI###
model_parallelcross.configure_traits()


# %%
view_probe1.captured[0].termination
formatter0 = EngFormatter(unit='m')
from_cm = 1e-2
mu_unicode = "\u03BC"
ray_shape = np.shape(view_probe1.captured[0].origin)
c_val = np.arange(0,ray_shape[0])
print(c_val)


fig, skewAx = plt.subplots(1,4,figsize=(32,9),constrained_layout=True)
fig.suptitle(f'Beam waist input number= {input_waist}{mu_unicode}m, wavelength input number = {input_wavelength}nm,\n Raypier input beam waist = {gausstest[0]}, Raypier input wavelength = {gausstest[1]:.4}\n Rayleigh range = {gausstest[2]:.6}cm, diffraction angle ={gausstest[3]:.5}',fontsize=20)


cf0 = skewAx[0].scatter(view_probe1.captured[0].origin[:,0]*1e-2
,view_probe1.captured[0].origin[:,1]*1e-2
,c=c_val,cmap="nipy_spectral"
)
skewAx[0].xaxis.set_major_formatter(formatter0)
skewAx[0].yaxis.set_major_formatter(formatter0)
skewAx[0].set_xlabel('x ',fontsize=15)
skewAx[0].set_ylabel('y ',fontsize=15)
skewAx[0].grid()
skewAx[0].set_title(f"@ Input skew rays\n Input waist ={input_waist}{mu_unicode}m",fontsize=20)
skewAx[0].tick_params(labelsize=15)

cf1 = skewAx[1].scatter(view_probe5.captured[0].termination[:,0]*1e-2
,view_probe5.captured[0].termination[:,1]*1e-2
,c=c_val,cmap="nipy_spectral"
)
skewAx[1].xaxis.set_major_formatter(formatter0)
skewAx[1].yaxis.set_major_formatter(formatter0)
skewAx[1].set_xlabel('x',fontsize=15)
skewAx[1].set_ylabel('y',fontsize=15)
skewAx[1].grid()
skewAx[1].set_title(f"@ Half rayleigh range\n Calc spot waist = {half_range:.5}{mu_unicode}m",fontsize=20)
skewAx[1].tick_params(labelsize=15)

cf2 = skewAx[2].scatter(view_probe10.captured[0].termination[:,0]*1e-2
,view_probe10.captured[0].termination[:,1]*1e-2
,c=c_val,cmap="nipy_spectral"
)
skewAx[2].xaxis.set_major_formatter(formatter0)
skewAx[2].yaxis.set_major_formatter(formatter0)
skewAx[2].set_xlabel('x ',fontsize=15)
skewAx[2].set_ylabel('y ',fontsize=15)
skewAx[2].grid()
skewAx[2].set_title(f"@ rayleigh range\n Calc spot waist = {full_range:.5}{mu_unicode}m",fontsize=20)
skewAx[2].tick_params(labelsize=15)

cf3 = skewAx[3].scatter(view_probe11.captured[0].termination[:,0]*1e-2
,view_probe11.captured[0].termination[:,1]*1e-2
,c=c_val,cmap="nipy_spectral"
)
skewAx[3].xaxis.set_major_formatter(formatter0)
skewAx[3].yaxis.set_major_formatter(formatter0)
skewAx[3].set_xlabel('x ',fontsize=15)
skewAx[3].set_ylabel('y ',fontsize=15)
skewAx[3].grid()
skewAx[3].set_title(f"@ 1.5 rayleigh range \n Calc spot waist = {past_range:.5}{mu_unicode}m",fontsize=20)
skewAx[3].tick_params(labelsize=15)

# %% Evolution of gaussian beam

z_vals = np.linspace(0,1.5*gausstest[2]) #cm
y_vals = np.sqrt(np.square(input_waist*1e-4)*(1+ np.square(z_vals/gausstest[2]))) #cm


fig, evolve = plt.subplots(1,1,figsize = [25,10])
fig.suptitle(f'Evolution of gaussian beam source beam traced in Raypier\n Beam waist = {input_waist}{mu_unicode}m, wavelength = {input_wavelength}nm,\n  Rayleigh range = {gausstest[2]:.4}cm, diffraction angle ={gausstest[3]:.5} ',fontsize=20)

evolve.grid()
evolve.tick_params(labelsize=15)

evolve.plot(z_vals*1e-2,y_vals*1e-2,c="#fd0202")

evolve.scatter(view_probe1.captured[0].origin[:,2]*1e-2
,view_probe1.captured[0].origin[:,1]*1e-2,c='m',label=f"Input - {input_waist}{mu_unicode}m waist")

evolve.scatter(view_probe1.captured[0].termination[:,2]*1e-2,view_probe1.captured[0].termination[:,1]*1e-2,c='#0170c8')

evolve.scatter(view_probe2.captured[0].termination[:,2]*1e-2,view_probe2.captured[0].termination[:,1]*1e-2,c='#0170c8')

evolve.scatter(view_probe3.captured[0].termination[:,2]*1e-2,view_probe3.captured[0].termination[:,1]*1e-2,c='#0170c8')

evolve.scatter(view_probe4.captured[0].termination[:,2]*1e-2,view_probe4.captured[0].termination[:,1]*1e-2,c='#0170c8')

evolve.scatter(view_probe5.captured[0].termination[:,2]*1e-2,view_probe5.captured[0].termination[:,1]*1e-2,c='#a333ff',label=f"Half rayleigh range - {half_range:.6}{mu_unicode}m waist")

evolve.scatter(view_probe6.captured[0].termination[:,2]*1e-2,view_probe6.captured[0].termination[:,1]*1e-2,c='#0170c8')

evolve.scatter(view_probe7.captured[0].termination[:,2]*1e-2,view_probe7.captured[0].termination[:,1]*1e-2,c='#0170c8')

evolve.scatter(view_probe8.captured[0].termination[:,2]*1e-2,view_probe8.captured[0].termination[:,1]*1e-2,c='#0170c8')

evolve.scatter(view_probe9.captured[0].termination[:,2]*1e-2,view_probe9.captured[0].termination[:,1]*1e-2,c='#0170c8')

evolve.scatter(view_probe10.captured[0].termination[:,2]*1e-2,view_probe10.captured[0].termination[:,1]*1e-2,c='#fc5e51',label=f"Rayleigh range - {full_range:.5}{mu_unicode}m waist")

evolve.scatter(view_probe11.captured[0].termination[:,2]*1e-2,view_probe11.captured[0].termination[:,1]*1e-2,c='#ff7815',label=f"1.5x Rayleigh range - {past_range:.5}{mu_unicode}m waist")

evolve.xaxis.set_major_formatter(formatter0)
evolve.yaxis.set_major_formatter(formatter0)

evolve.legend(loc='upper left',fontsize=17)

evolve.set_xlabel('z ',fontsize=20)
evolve.set_ylabel('y ',fontsize=20)

# %%
