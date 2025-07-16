# %% 
import finesse
finesse.init_plotting()
from finesse_ligo.factory import ALIGOFactory, CARMFactory
from finesse_ligo.actions import InitialLockLIGO, DARM_RF_to_DC
import finesse_ligo
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from finesse.ligo.maps import (get_test_mass_surface_profile_interpolated, 
                               aligo_O4_TM_aperture, aligo_O4_ESD_inner_aperture)
from finesse.knm.maps import Map
# from finesse.detectors import FieldDetector, AmplitudeDetector
import finesse.analysis.actions as fa
import finesse.components as fc
from finesse.plotting.plot import get_2d_field
from scipy.signal import find_peaks
from tqdm import tqdm
import matplotlib as mpl
from scipy.interpolate import CubicSpline
mpl.rcParams.update({'text.usetex': False,
                     'mathtext.fontset': 'cm',
                     'lines.linewidth': 2,
                     'lines.markersize': 10,
                     'font.size': 15,
                     'axes.grid': True,
                     'grid.alpha': 0.3,
                     'legend.loc': 'best',
                     'figure.dpi': 100, 
                     'pdf.compression': 9,
                     'xtick.labelsize' : 15,
                     'ytick.labelsize' : 15})
plt.style.use('dark_background')
from tabulate import tabulate
from scipy.interpolate import splrep, BSpline
from scipy.signal import butter, lfilter, medfilt
import h5py
import hom_frequency_response_functions as hom_fr
import matplotlib.animation as animation
import h5py

# %%
# Get RH thermal data:
rh_data = {}
rh_data["itm_tl_file"] = "./thermal_data/itm_opd_rh_1W_2da.txt"
rh_data["cp_tl_file"] = "./thermal_data/cp_opd_rh_1W_2da.txt"
rh_data["itm_td_file"] = "./thermal_data/itm_deform_rh_1W_2da.txt"
rh_data["etm_td_file"] = "./thermal_data/etm_deform_rh_1W_2da.txt"

hom_fr.get_thermal_data(rh_data)
# %%
# Get absorptiong fem data
time = np.hstack((np.arange(0,1820,20), np.arange(1840,3640,40, ),
                  np.arange(3660, 7260, 60), np.arange(7320, 18120, 120),
                  np.arange(18300, 36300, 300)))
absorption_data = {}
# X-arm
absorption_data["itmx_tl_file"] = "./thermal_data/itmx_tl_llo_power_up.txt"
absorption_data["cpx_tl_file"] = "./thermal_data/cpx_tl_llo_power_up.txt"
absorption_data["itmx_td_file"] = "./thermal_data/itmx_td_llo_power_up.txt"
absorption_data["etmx_td_file"] = "./thermal_data/etmx_td_llo_power_up.txt"

# Y-arm
absorption_data["itmy_tl_file"] = "./thermal_data/itmy_tl_llo_power_up.txt"
absorption_data["cpy_tl_file"] = "./thermal_data/cpy_tl_llo_power_up.txt"
absorption_data["itmy_td_file"] = "./thermal_data/itmy_td_llo_power_up.txt"
absorption_data["etmy_td_file"] = "./thermal_data/etmy_td_llo_power_up.txt"

def get_absortion_data(time, absorption_data):
    Nt = len(time)
    rr = np.sqrt(hom_fr.X ** 2 + hom_fr.Y ** 2)
    for key in tqdm(list(absorption_data.keys())):
        new_key = key.split("_file")[0]  
        absorption_data[new_key]  = []
        data = np.loadtxt(absorption_data[key], comments = "%")
        _r = np.sort(data[:,0])
        _opds = data[:,2:]
        assert _opds.shape[-1] == Nt
        
        for i in range(Nt):
            opd_func = CubicSpline(_r, _opds[:,i][np.argsort(data[:,0])])
            absorption_data[new_key].append(np.nan_to_num(opd_func(rr)))

get_absortion_data(time, absorption_data)

a0_itmx = 0.4
a1_itmx = 0.75
a0_itmy = 0.3
a1_itmy = 0.82
a0_etmx = 0.32
a1_etmx = 0.97
a0_etmy = 0.14
a1_etmy = 0.73 

absorption_data["itmx_td"] = (np.asarray(absorption_data["itmx_td"]) * (a1_itmx / a0_itmx ))[::2]
absorption_data["itmy_td"] = (np.asarray(absorption_data["itmy_td"]) * (a1_itmy / a0_itmy ))[::2]
absorption_data["etmx_td"] = (np.asarray(absorption_data["etmx_td"]) * (a1_etmx / a0_etmx ))[::2]
absorption_data["etmy_td"] = (np.asarray(absorption_data["etmy_td"]) * (a1_etmy / a0_etmy ))[::2]
absorption_data["itmx_tl"] = (np.asarray(absorption_data["itmx_tl"]) * (a1_itmx / a0_itmx ))[::2]
absorption_data["itmy_tl"] = (np.asarray(absorption_data["itmy_tl"]) * (a1_itmy / a0_itmy ))[::2]
absorption_data["cpx_tl"] =  (np.asarray(absorption_data["cpx_tl"] ) * (a1_itmx / a0_itmx ))[::2]
absorption_data["cpy_tl"] =  (np.asarray(absorption_data["cpy_tl"] ) * (a1_itmy / a0_itmy ))[::2]
time = time[::2]
# abs_maps["itmx_tl"] = (absorption_data["itmx_tl"] + absorption_data["cpx_tl"]) * (a1_itmx / a0_itmx )
# abs_maps["itmy_tl"] = (absorption_data["itmy_tl"] + absorption_data["cpy_tl"]) * (a1_itmy / a0_itmy )

# %%
# Build cold state with RH:
factory = ALIGOFactory(Path("C:/Users/cao/Documents/ligo-commissioning-modeling") / "LLO" / "yaml" / "llo_O4.yaml")
factory.params.INPUT.LASER.power = 2 
factory.options.LSC.add_output_detectors = True
factory.options.ASC.add = False
factory.options.thermal.add = True
llo = factory.make()
hom_fr.get_cold_maps(llo, factory=factory)
llo.beam_trace()
llo_base = llo.deepcopy()
initial_parameters = {p.full_name: p.value for p in llo.all_parameters}

# %%
llo_cold_rh = llo_base.deepcopy()
llo_cold_rh.modes(maxtem=6)
lock = InitialLockLIGO(
    exception_on_lock_fail=False,
    lock_steps=5000,
    gain_scale=0.5,
    pseudo_lock_arms=False,
    run_locks=True,
)
sol_lock_rh = llo_cold_rh.run(
        lock
)
# Draging lock to see whether we can improve operation point
fractions = np.linspace(0, 0.60, 60)
P_RHs = {} 
P_RHs["ITMX"] = 2 * 0.5
P_RHs["ETMX"] = 2 * 1.1
P_RHs["ITMY"] = 2 * 0.4
P_RHs["ETMY"] = 2 * 1.2
outs = []
for _f in tqdm(fractions):
    hom_fr.get_rh_ss_maps(llo_cold_rh, 
                          initial_parameters = initial_parameters,
                          RH_dict=rh_data, 
                          P_RH_dict=P_RHs,
                          itm_efficiency=_f,
                          etm_efficiency=_f)
    _sol = llo_cold_rh.run(
        fa.Series(
            fa.RunLocks(method="proportional",
                        max_iterations=2000,
                        display_progress=True,
                        exception_on_fail=False),
            fa.Noxaxis()
        )
    )
    llo_cold_rh.run(fa.SetLockGains(gain_scale=0.4))
    outs.append(_sol['noxaxis'])

# llo_cold_rh.run(
#     DARM_RF_to_DC())
# %%
# Get cold ring heater parameter:
cold_rh_params = {}
for TM in (llo_cold_rh.ITMX, llo_cold_rh.ETMX,
           llo_cold_rh.ITMY, llo_cold_rh.ETMY):
    cold_rh_params[TM.Rcx.full_name] =  TM.Rcx.value

    cold_rh_params[TM.name + "_opd"] =  TM.surface_map.opd.copy()
for ITMlen in (llo_cold_rh.ITMXlens, llo_cold_rh.ITMYlens):
    cold_rh_params[ITMlen.f.full_name] =  ITMlen.f.value
    cold_rh_params[ITMlen.name + "_opd"] =  ITMlen.OPD_map.opd.copy()
llo_power_up = llo_cold_rh.deepcopy()
llo_power_up.beam_trace()
# %%
# Run pseudo-power up 
hom1_fr_dat = []
hom2_fr_dat = []
models = []
sols_out = []
for i, t in (enumerate(time)):
    print(f"time: {t} s")
    itmx_map = absorption_data["itmx_td"][i]
    itmy_map = absorption_data["itmy_td"][i]
    etmx_map = absorption_data["etmx_td"][i]
    etmy_map = absorption_data["etmy_td"][i]
    itmxlen_map = absorption_data["itmx_tl"][i] + absorption_data["cpx_tl"][i] 
    itmylen_map = absorption_data["itmy_tl"][i] + absorption_data["cpy_tl"][i] 

    # Update surface deformation
    for TM, TM_map in zip((llo_power_up.ITMX, llo_power_up.ETMX, 
                           llo_power_up.ITMY, llo_power_up.ETMY),
                           (itmx_map, etmx_map,itmy_map, etmy_map)):
        TM.surface_map = Map(hom_fr.x, hom_fr.y,
                             amplitude = hom_fr.TM_aperture,
                             opd = cold_rh_params[TM.name + "_opd"] + TM_map
                             )
        spot_size = TM.p1.i.qx.w
        a = TM.surface_map.get_radius_of_curvature_reflection(spot_size)
        TM.Rc = 2 / (2 / float(cold_rh_params[TM.Rcx.full_name]) + 2 / np.asarray(a))
        TM.surface_map.remove_curvatures(spot_size)
        TM.surface_map.remove_tilts(spot_size)
        print(f"{TM.name} Roc: {TM.Rc}")
    
    # Update thermal lens
    for TMlen, TMlen_map in zip((llo_power_up.ITMXlens, llo_power_up.ITMYlens),
                                (itmxlen_map, itmylen_map)):
        TMlen.OPD_map = Map(hom_fr.x, hom_fr.y,
                            amplitude = hom_fr.X_aperture,
                            opd = cold_rh_params[TMlen.name + "_opd"] + TMlen_map
                            )
        spot_size = TMlen.p1.i.qx.w
        a = TMlen.OPD_map.get_thin_lens_f(
            spot_size, average=True
        )
        TMlen.f.value = 1 / (1 / float(cold_rh_params[TMlen.f.full_name]) + 1/a)
        TMlen.OPD_map.remove_curvatures(
            spot_size, mode="average"
        )
        TMlen.OPD_map.remove_tilts(spot_size)
        print(f"{TMlen.name} focal length: {TMlen.f.value}")

    # Locking 
    _sol = llo_power_up.run(
        fa.Series(
            fa.RunLocks(method="proportional",
                        max_iterations=2000,
                        display_progress=True,
                        exception_on_fail=False),
            fa.Noxaxis()
        )
    )
    sols_out.append(_sol['noxaxis'])
    models.append(llo_power_up.deepcopy())

    # Frequency Response
    print("Run frequency response ...")
    _hom1_fr, _hom2_fr = hom_fr.get_hom_frequency_drive(llo_power_up)

    hom1_fr_dat.append(_hom1_fr)
    hom2_fr_dat.append(_hom2_fr)

    llo_power_up.run(fa.SetLockGains(gain_scale=0.4))
    llo_power_up.beam_trace()

# %%
from pathlib import Path
# import gpstime
import pickle
import dill

path = Path("./data/")
# data_time = int(gpstime.gpsnow())
path.mkdir(exist_ok=True)
out = path / f"llo_hom_fshift_powerup_00.pkl"
print(out)
with open(out, "wb") as file:
    dill.dump(
        {
            "parameters": factory.params,
            "options": factory.options,
            "outs": sols_out,
            "t": time,
            "hom1_freq_response": hom1_fr_dat,
            "hom2_freq_response": hom2_fr_dat,
            # "models": [m.unparse() for m in models],
        },
        file,
    )
# %%
id = -1
model = models[id]
model.beam_trace()
hom2fr_sum  = hom_fr.compute_sum_amplitude(model, hom2_fr_dat[id])
fig, axs = hom_fr.plot_freq_response(hom2_fr_dat[id], hom2fr_sum )
# Get 00 component:
ports = ("OMC", "TR-Y", "TR-X")
for  _p, ax in zip(ports, axs):
    E00_fr = hom2_fr_dat[id][_p]["FrequencyResponse"][:,0]

    # Adding field
    ax.semilogy(hom2_fr_dat[i]["freq"]/1e3, abs(E00_fr))
# %%
# Getting real data:
# arm power:
Parm_file = "./thermal_data/llo_arm_power.txt"
Parm_dat =  np.loadtxt(Parm_file)
plot_dat_llo_file = './data/llo_power_up_ani_1381876218.h5'
llo_data = {}
with h5py.File(plot_dat_llo_file, "r") as f:
    llo_data_keys = list(f.keys())
    for key in llo_data_keys:
        dset = f[key]
        if isinstance(dset, h5py.Dataset):
            llo_data[key] = f[key][()]
        else:
            llo_data[key] = {}
            gkeys =  list(f[key].keys())
            for _gk in gkeys:
                gdset = dset[_gk]
                if isinstance(gdset, h5py.Dataset):
                    llo_data[key][_gk] = gdset[()]
                else:
                    llo_data[key][_gk] = {}
                    sgkeys = list(gdset.keys())
                    for _sgk in sgkeys:
                        sgdset = gdset[_sgk]
                        llo_data[key][_gk][_sgk] = sgdset[()]
llo_psd_processed = llo_data['LG1mode']['PSD']['processed']
llo_psd_raw = llo_data['LG1mode']['PSD']['raw']
llo_psd_f = llo_data['LG1mode']['PSD']['frequency']
llo_psd_time = llo_data['LG1mode']['PSD']['time']

# %%
# Get mode mismatch from PRC --> ARM
# cavPRX(x,y) --> cavXARM(x,y)
# cavPRY(x,y) --> cavYARM(x,y)
# cavXARM(x,y) --> cavYARM(x,y)
cavs1 = ["cavPRX", "cavPRY", "cavSRX", "cavSRY", "cavXARM"]
cavs2 = ["cavXARM", "cavYARM","cavXARM", "cavYARM", "cavYARM"]

mm_data = {}
for _model in tqdm(models):
    for _cav1, _cav2 in zip(cavs1, cavs2):
        c_c = _cav1+'-'+_cav2
        if c_c not in mm_data:
            mm_data[c_c] =  []
        mm_data[c_c].append(np.asarray(_model.cavity_mismatch(_cav1, _cav2)))

for key in list(mm_data.keys()):
    mm_data[key] = np.asarray(mm_data[key])

# %%
# Plotting mode-mismatch
fig, axs = plt.subplots(2,1, figsize=(8,10), sharex = True)
cav_keys = list(mm_data.keys())
titles = ("tangential", "saggital")
for i, ax in enumerate(axs):
    for ck in cav_keys:
        ax.plot(time/3600, 100* mm_data[ck][:,i],label=ck)
    ax.legend(loc=3)
    ax.set_yscale('log')
    ax.set_xscale("log")
    ax.set_ylabel("Mode mismatch [%]")
    ax.set_title(titles[i],fontsize=15)
axs[1].set_xlabel("Time [hr]")
# %%
# Plotting OMC:
def plot_OMC(hom2_sol, plot_modes = True):
    fig, axs = plt.subplots(3,1, figsize=(8,12)) 
    lines_sim = {}
    lines_dat = {}
    
    # plot 1:
    axs[0].plot(Parm_dat[:,0]/3600, Parm_dat[:,1]/1e3) 
    axs[0].set_xlabel("Time [hr]")
    axs[0].set_ylabel("Power [kW]")
    ax0ylim = axs[0].get_ylim()
    tline, = axs[0].plot((0,0), ax0ylim, 'gray', lw=1)



    # plot2:
    # find time:
    it = 0
    psd_idx = np.argmin(abs(llo_psd_time - it))
    lines_dat["raw"], = axs[1].semilogy(llo_psd_f/1e3, llo_psd_raw[psd_idx], 'C2', alpha=0.3 )
    lines_dat["processed"], = axs[1].semilogy(llo_psd_f/1e3, llo_psd_processed[psd_idx], 'C2' )
    axs[1].set_xlim(9.6, 10.9)
    axs[1].set_xlabel("Frequency [kHz]")
    axs[1].set_ylabel("PSD [a.u.]")
    axs[1].set_ylim(6e-5, 2e2)
    # plot3

    freq = hom2_sol['freq']
    port = "OMC"
    omc_sol = hom2_sol[port]["FrequencyResponse"]
    mode_idx  = [0, 3,4,5]
    mode_names = ["HG00", "HG20", "HG11", "HG02"]
    # Compute sum of power of all modes:
    psum =  np.sqrt((abs(omc_sol)**2).sum(axis=1))
    lines_sim["sum"], = axs[2].semilogy(freq/1e3, psum, 'C1', label = "Sum")
    if plot_modes:
        for id, name in zip(mode_idx, mode_names):
            lines_sim[name], = axs[2].semilogy(freq/1e3, abs(omc_sol[:, id]) , '--', label = name)
        axs[2].legend(loc=4)
    axs[2].set_xlabel("Frequency [kHz]")
    axs[2].set_ylabel('Magnitude [W/Hz]')
    axs[2].set_xlim(9.6, 10.9)
    fig.tight_layout()
    return fig, axs, lines_sim, lines_dat, tline

# plotting all ports and mm:
def plot_all_ports_hom(hom_sol, mm_sol, plot_modes = True, hom_order=2):
    """
    Plotting all ports, and mode mismatch

    Parameters:
    -----------------------
    hom2_sol:       dict
                    Second order mode solution
    mm_sol  :       dict
                    mode mismatch arrays
    plot_modes:     boolean
                    plot modes contribution

    Returns:
    ------------------------
    fig, axs:   figure and axes of the figure
    tlines:     t
    """
    fig, axs =plt.subplots(3,2, figsize = (12, 14) )
    
    tlines = {}
    # First column, j=0
    # Power
    axs[0,0].plot(Parm_dat[:,0]/3600, Parm_dat[:,1]/1e3) 
    axs[0,0].set_ylabel("Power [kW]")
    axs[0,0].set_xscale("log")
    axs[0,0].set_xlim(20/3600, 10)
    ax0ylim = axs[0,0].get_ylim()
    tlines["power"], = axs[0,0].plot((0,0), ax0ylim, 'gray', lw=1)

    # Mode-mismatches:
    axes = (axs[1,0], axs[2,0])
    cav_keys = list(mm_sol.keys())
    titles = ("tangential", "saggital")
    for ia, ax in enumerate(axes):         
        for ck in cav_keys:
            ax.plot(time/3600, 100* mm_sol[ck][:,ia],label=ck)
        ax.legend(loc=3)
        ax.set_yscale('log')
        ax.set_xscale("log")
        ax.set_ylabel("Mode mismatch [%]")
        ax.set_title(titles[ia],fontsize=15)
        _ylim = ax.get_ylim()
        tlines["mm"+titles[ia]], = ax.plot((0,0), _ylim, 'gray', lw=1)
        ax.set_xlim(20/3600, 10)
    axs[2,0].set_xlabel("Time [hr]")

    # Second column
    lines_sim = {}
    ports = ("OMC", "TR-X", "TR-Y")
    freq = hom_sol['freq']
    if hom_order == 2:
        mode_idx  = [0,3,4,5]
        mode_names = ["HG00", "HG20", "HG11", "HG02"]
    elif hom_order == 1:
        mode_idx  = [0,1,2]
        mode_names = ["HG00", "HG10", "HG01"]
    for ip, _port in enumerate(ports):
        psol = hom_sol[_port]["FrequencyResponse"]
        psum =  np.sqrt((abs(psol)**2).sum(axis=1))
        lines_sim["sum_"+ _port], = axs[ip, 1].semilogy(freq/1e3, psum, 'C1', label = "Sum")
        if plot_modes:
            for id, name in zip(mode_idx, mode_names):
                lines_sim[name+'_'+_port], = axs[ip, 1].semilogy(freq/1e3, abs(psol[:, id]) , '--', label = name)
        axs[ip,1].legend(loc=4)
        axs[ip,1].set_ylabel(r'Magnitude [$\mathrm{\sqrt{W}}$/Hz]')
        axs[ip,1].set_title(_port)
    axs[2,1].set_xlabel("Frequency [kHz]")
    fig.tight_layout()
    return fig, axs, tlines, lines_sim

def compute_sum_all_ports(hom_sol,hom_order=2 ):
    ports = ("OMC", "TR-X", "TR-Y")
    ports_dat = {}
    for sol in hom_sol:
        for port in ports:
            if port not in ports_dat:
                ports_dat[port] = {}
            psol = sol[port]["FrequencyResponse"]
            psum =  np.sqrt((abs(psol)**2).sum(axis=1))
            if "sum" not in ports_dat[port]:
                ports_dat[port]["sum"] = []
            ports_dat[port]["sum"].append(psum)

            if hom_order == 2:
                mode_idx  = [0, 3,4,5]
                mode_names = ["HG00", "HG20", "HG11", "HG02"]
            elif hom_order == 1:
                mode_idx  = [0, 1, 2]
                mode_names = ["HG00", "HG10", "HG01"]
            for idx, mode in  zip(mode_idx, mode_names):
                if mode not in ports_dat[port]:
                    ports_dat[port][mode] = []
                ports_dat[port][mode].append(abs(psol[:,idx]))
    return ports_dat
# %%
# Compute sums at OMC
omc_data = {}
omc_data["sum"]=[]
for _fr in hom2_fr_dat:
    port = "OMC"
    omc_sol = _fr[port]["FrequencyResponse"]
    psum =  np.sqrt((abs(omc_sol)**2).sum(axis=1))
    mode_idx  = [0, 3,4,5]
    mode_names = ["HG00", "HG20", "HG11", "HG02"]
    omc_data["sum"].append(psum)
    for idx, mode in  zip(mode_idx, mode_names):
        if mode not in omc_data:
            omc_data[mode] = []
        omc_data[mode].append(abs(omc_sol[:,idx]))
# %%
# generate animation to compare OMC transfer function with real data
fig, axs, lines_sim, lines_dat, tline = plot_OMC(hom2_fr_dat[-1], plot_modes =False)

ax0ylim = axs[0].get_ylim()
freq = hom2_fr_dat[0]['freq']
def update(idx, freq, data, traces_sim, traces_dat, tline):
    tline.set_data((time[idx]/3600,time[idx]/3600), ax0ylim)
    psd_idx = np.argmin(abs(llo_psd_time - time[idx]))
    traces_dat["raw"].set_data(llo_psd_f/1e3, llo_psd_raw[psd_idx])
    traces_dat["processed"].set_data(llo_psd_f/1e3, llo_psd_processed[psd_idx])
    traces_names = list(traces_sim.keys())
    for _trn in traces_names:
        traces_sim[_trn].set_data(freq, data[_trn][idx])
    return  traces_sim, traces_dat, tline

ani = animation.FuncAnimation(fig, update, len(time), interval=100,
                              fargs = [freq/1e3, omc_data, lines_sim, lines_dat, tline ])
writervideo = animation.FFMpegWriter(fps=5) 
ani.save('./figures/omc_hom_thermalise.mp4',writer=writervideo)

# %%
# Compute all ports traces to be plot
ports_dat_10kHz = compute_sum_all_ports(hom2_fr_dat) 
 
# %%
fig, axs, tlines, lines_sim = plot_all_ports_hom(hom2_fr_dat[-1], mm_data, plot_modes = True)
freq = hom2_fr_dat[0]['freq']
ylim_tlines = {}
tkeys = list(tlines.keys())
for k, _tk in enumerate(tkeys): 
    ylim_tlines[_tk] = axs[k,0].get_ylim()
def update_allports(idx, freq, data,  traces_time, traces_dat):
    tkeys = list(traces_time.keys())
    for _tk in (tkeys):
        traces_time[_tk].set_data((time[idx]/3600,time[idx]/3600),ylim_tlines[_tk] )
    dat_key = list(traces_dat.keys())
    for dkey in dat_key:
        _mode, _port = dkey.split("_")
        traces_dat[dkey].set_data(freq, data[_port][_mode][idx])
    return traces_time, traces_dat
ani = animation.FuncAnimation(fig, update_allports, len(time), interval=100,
                              fargs = [freq/1e3, ports_dat_10kHz, 
                                       tlines, lines_sim ])
writervideo = animation.FFMpegWriter(fps=5) 
ani.save('./figures/allports_10kHz_thermalise_model.mp4',writer=writervideo)

# %%
# Get 5 kHz data
ports_dat_5kHz = compute_sum_all_ports(hom1_fr_dat, hom_order=1)

# %%
# Generate movive for 5kHz ( firts order mode)
fig, axs, tlines, lines_sim = plot_all_ports_hom(hom1_fr_dat[0], mm_data, plot_modes = True, hom_order=1)
freq = hom1_fr_dat[0]['freq']
ani = animation.FuncAnimation(fig, update_allports, len(time), interval=100,
                              fargs = [freq/1e3, ports_dat_5kHz, 
                                       tlines, lines_sim ])
writervideo = animation.FFMpegWriter(fps=5) 
ani.save('./figures/allports_5kHz_thermalise_model.mp4',writer=writervideo)


# %%
# Get frequency response at higher FSRs
F_Hz_HOM1_FSR1 = np.linspace(4.5e3, 5.5e3, 200) + llo.cavYARM.FSR
F_Hz_HOM1_FSR2 = np.linspace(4.5e3, 5.5e3, 200) + 2 * llo.cavYARM.FSR
a_tf_fsr1 = {}
a_tf_fsr2 = {}
a_tf_fsr1['freq'] = F_Hz_HOM1_FSR1
a_tf_fsr2['freq'] = F_Hz_HOM1_FSR2
a_tf_fsr1["TR_Y"]= []
a_tf_fsr2["TR_Y"]= []

for _model in tqdm(models):
    _model.fsig.f = 1
    fsig = _model.fsig.f.ref
    out_hom1_FSRs = _model.run(
        fa.Series(
            fa.FrequencyResponse4(F_Hz_HOM1_FSR1, 
                                [_model.L0.frq],
                                [(_model.ETMY.p1.o, -fsig)],
                                name='hom1_fsr1')
                                ,
            fa.FrequencyResponse4(F_Hz_HOM1_FSR2, 
                                [_model.L0.frq],
                                [(_model.ETMY.p1.o, -fsig)],
                                name='hom1_fsr2')
                                )
    )
    # Process FSR1:
    
    a_tf_fsr1["TR_Y"].append(out_hom1_FSRs['hom1_fsr1'].out[:,0,0,:])
    a_tf_fsr2["TR_Y"].append(out_hom1_FSRs['hom1_fsr2'].out[:,0,0,:])
    
# %%
a_tf_fsr0  = {}
a_tf_fsr0["freq"] = hom1_fr_dat[0]['freq']
a_tf_fsr0["TR_Y"] = []
for dat in hom1_fr_dat:
    a_tf_fsr0["TR_Y"].append(dat["TR-Y"]["FrequencyResponse"])  

# Plot time evolution of HOM1 at different FSRs:
def plot_hom1_FSRs(fsr0_dat, fsr1_dat, fsr2_dat, idx = 0, plot_modes=True):
    """
    # plotting first order mode atmultiple FSRs
    
    """
    fsr_dats =  (fsr0_dat, fsr1_dat, fsr2_dat)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    names = ("FSR0", "FSR1", "FSR2")
    traces = {}
    hom_idx = (1, 2)
    hom_names = ("HG10", "HG01")
    for fsr_dat, name, ax in zip(fsr_dats, names, axs):
        freq = fsr_dat["freq"]
        fr_dat = fsr_dat["TR_Y"][idx]

        # Cumpute sum:
        fr_sum =  np.sqrt(np.sum(abs(fr_dat[:, 1:3]) ** 2, axis=1 ))
        traces[name + "_sum"], =  ax.plot(freq / 1e3,fr_sum, label = "HG1 Sum" )

        if  plot_modes:
            for mode, _midx in zip( hom_names, hom_idx):
                traces[name + "_" + mode], = ax.plot(freq /1e3,  abs(fr_dat[:,_midx]), '--', label = mode) 
        ax.legend(loc=4)
        ax.set_xlabel("Frequency [kHz]")
        ax.set_ylabel( r"Magnitude [$\mathrm{\sqrt{W}}$/Hz]")
        ax.set_yscale("log")
    fig.tight_layout()
    return fig, axs, traces

# %%
fig, axs, hom1_traces = plot_hom1_FSRs(a_tf_fsr0, a_tf_fsr1, a_tf_fsr2, idx=-1)
hom1_plot_data = {}
hom1_keys = list(hom1_traces.keys())
for _k in hom1_keys:
    hom1_plot_data[_k] = []

fsr_keys = ("FSR0", "FSR1", "FSR2")
hom_keys = ("sum", "HG10", "HG01")

for fsr_dat, fsr_k in tqdm(zip((a_tf_fsr0, a_tf_fsr1, a_tf_fsr2),
                          fsr_keys)):
    for dat in fsr_dat["TR_Y"]:
        hom1_plot_data[fsr_k + "_sum"].append(np.sqrt
                                              (np.sum(abs(dat[:, 1:3]) ** 2, axis=1 )))
        hom1_plot_data[fsr_k + "_HG10"].append((abs(dat[:, 1])))
        hom1_plot_data[fsr_k + "_HG01"].append((abs(dat[:, 2])))

# %%
freqs_fsr = {}
freqs_fsr["FSR0"]=a_tf_fsr0["freq"]
freqs_fsr["FSR1"]=a_tf_fsr1["freq"]
freqs_fsr["FSR2"]=a_tf_fsr2["freq"]

def update_fsrs(idx, freq_dict, data_dict, traces_dict):
    traces_keys = list(traces_dict.keys())
    freq_keys = list(freq_dict.keys())
    for tkey in traces_keys:
        for fkey in freq_keys:
            if fkey in tkey:
                traces_dict[tkey].set_data(freq_dict[fkey]/1e3, data_dict[tkey][idx])
    return traces_dict

# %%
# Generate animatoion for HOM1:
fig, axs, hom1_traces = plot_hom1_FSRs(a_tf_fsr0, a_tf_fsr1, a_tf_fsr2, idx=-1)
ani = animation.FuncAnimation(fig, update_fsrs, len(time), interval=100,
                              fargs = [freqs_fsr, hom1_plot_data, hom1_traces ])
writervideo = animation.FFMpegWriter(fps=5) 
ani.save('./figures/tr_y_hom1_thermalise_model_fsrs.mp4',writer=writervideo)
# %%
# Test steady state first to check if it's ok
def get_sh_ss_maps(model, initial_parameters, absorption_map_dict, fraction = 0.1):
    model.beam_trace()
    # optics_names = list(absorption_map_dict.keys())
    optics = (model.ITMX, model.ETMX, model.ITMY, model.ETMY)

    # Apply RH deformation to test mass surfaces

    td_itmx = absorption_map_dict["itmx_td"]
    td_etmx = absorption_map_dict["etmx_td"]
    td_itmy = absorption_map_dict["itmy_td"]
    td_etmy = absorption_map_dict["etmy_td"]
    tds = (td_itmx, td_etmx, td_itmy, td_etmy)
    tl_itmx =  absorption_map_dict["itmx_tl"]
    tl_itmy =  absorption_map_dict["itmy_tl"]
    tls = (tl_itmx, tl_itmy)
    for TM, _map in zip(optics, tds):
        _x = TM.surface_map.x
        _y = TM.surface_map.y
        _mask = TM.surface_map.amplitude.copy()
        TM.surface_map = Map(
            _x,
            _y,
            amplitude= _mask,
            opd=initial_parameters[TM.name + "_opd"] + fraction * _map
        )
        spot_size = TM.p1.i.qx.w
        a = TM.surface_map.get_radius_of_curvature_reflection(spot_size)
        TM.Rc = 2 / (2 / float(initial_parameters[TM.Rcx.full_name]) + 2 / np.asarray(a))
        TM.surface_map.remove_curvatures(spot_size)
        TM.surface_map.remove_tilts(spot_size)
        print(f"{TM.name} Roc: {TM.Rc}")

    lens = (model.ITMXlens, model.ITMYlens)

    for ITM, _map in zip(lens, tls):
        _mask = ITM.OPD_map.amplitude.copy()
        _tl_map = (initial_parameters[ITM.name + "_opd"] + fraction * _map)
        ITM.OPD_map = Map(
            _x,
            _y,
            amplitude = _mask,
            opd = _tl_map
        )
        spot_size = ITM.p1.i.qx.w
        a = ITM.OPD_map.get_thin_lens_f(
            spot_size, average=True
        )
        ITM.f.value = 1 / (1 / float(initial_parameters[ITM.f.full_name]) + 1/a)
        ITM.OPD_map.remove_curvatures(
            spot_size, mode="average"
        )
        ITM.OPD_map.remove_tilts(spot_size)
        print(f"{ITM.name} focal length: {ITM.f.value}")
    model.beam_trace()

# %%
# Try inrcreasing absorption:
i = -1 
a0_itmx = 0.4
a1_itmx = 0.75
a0_itmy = 0.3
a1_itmy = 0.82
a0_etmx = 0.32
a1_etmx = 0.97
a0_etmy = 0.14
a1_etmy = 0.73 

abs_maps = {}
abs_maps["itmx_td"] = absorption_data["itmx_td"][i] * (a1_itmx / a0_itmx )
abs_maps["itmy_td"] = absorption_data["itmy_td"][i] * (a1_itmy / a0_itmy )
abs_maps["etmx_td"] = absorption_data["etmx_td"][i] * (a1_etmx / a0_etmx )
abs_maps["etmy_td"] = absorption_data["etmy_td"][i] * (a1_etmy / a0_etmy )
abs_maps["itmx_tl"] = (absorption_data["itmx_tl"][i] + absorption_data["cpx_tl"][i]) * (a1_itmx / a0_itmx )
abs_maps["itmy_tl"] = (absorption_data["itmy_tl"][i] + absorption_data["cpy_tl"][i]) * (a1_itmy / a0_itmy )

llo_ss = llo_cold_rh.deepcopy()
llo_ss.beam_trace()

# %%
fractions = np.linspace(0,1,50)
for _f in tqdm(fractions):
    get_sh_ss_maps(llo_ss,initial_parameters = cold_rh_params, 
                   absorption_map_dict=abs_maps, fraction=_f)
    _sol = llo_ss.run(
        fa.Series(
            fa.RunLocks(method="proportional",
                        max_iterations=3000,
                        display_progress=True,
                        exception_on_fail=False),
            fa.Noxaxis()
        )
    )
    llo_ss.run(fa.SetLockGains(gain_scale=0.4))
    # outs.append(_sol['noxaxis'])

# %%
hom1_fr_ss, hom2_fr_ss = hom_fr.get_hom_frequency_drive(llo_ss)
hom2fr_sum_ss  = hom_fr.compute_sum_amplitude(llo_ss, hom2_fr_ss)

# %%
fig, axs = hom_fr.plot_freq_response(hom2_fr_ss, hom2fr_sum_ss )
for ax in axs:
    ylim = ax.set_ylim()
    ax.set_ylim( ylim[1]/4e3, ylim[1])
# %%
hom1fr_sum_ss  = hom_fr.compute_sum_amplitude(llo_ss, hom1_fr_ss)

# %%
fig, axs = hom_fr.plot_freq_response(hom1_fr_ss, hom1fr_sum_ss )

# %%
