# %%
import numpy as np
import finesse
finesse.init_plotting()
from scipy.interpolate import CubicSpline
from tabulate import tabulate
from scipy.interpolate import splrep, BSpline
from scipy.signal import butter, lfilter, medfilt
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

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

# %%
# Files from O4b:
llo_powerup_file_O4b = "./llo_power_up_data_1398542418.hdf5"
llo_psd_O4b = "./llo_omc_dcpd_1398542418_50mHz.pkl"

llo_powerup_data_O4b = {}
f =  h5py.File(llo_powerup_file_O4b, "r") 
chans =list(f.keys())
for chan in chans:
    if chan == "L1:OMC-PI_DCPD_64KHZ_AHF_DQ":
        continue
    else:
        llo_powerup_data_O4b[chan] = f[chan][()]

llo_powerup_data_O4b["time"] = np.linspace(0, 3600 * 10 , 3600 * 10 *16)

# Data from O4a:
llo_powerup_file_O4a = "./llo_power_up_data_1381876218.hdf5"
llo_powerup_data_O4a = {}
f =  h5py.File(llo_powerup_file_O4a, "r") 
chans =list(f.keys())
for chan in chans:
    if chan == "L1:OMC-PI_DCPD_64KHZ_AHF_DQ":
        continue
    else:
        llo_powerup_data_O4a[chan] = f[chan][()]
llo_powerup_data_O4a["time"] = np.linspace(0, 3600 * 10 , 3600 * 10 *16)


# %%
# Powers in O4b:
llo_powerup_data_O4b["PRG"] = 0.92 * llo_powerup_data_O4b["L1:LSC-POP_A_LF_OUTPUT"] / llo_powerup_data_O4b["L1:IMC-IM4_TRANS_SUM_OUTPUT"]
llo_powerup_data_O4b['9PRG'] = 0.0415 * llo_powerup_data_O4b['L1:LSC-POPAIR_B_RF18_I_MON'] / llo_powerup_data_O4b['L1:IMC-IM4_TRANS_SUM_OUTPUT']
llo_powerup_data_O4b['Parm'] = llo_powerup_data_O4b['L1:IMC-IM4_TRANS_SUM_OUTPUT'] * llo_powerup_data_O4b['PRG'] / 2 * 265

llo_powerup_data_O4a["PRG"] = 0.92 * llo_powerup_data_O4a["L1:LSC-POP_A_LF_OUTPUT"] / llo_powerup_data_O4a["L1:IMC-IM4_TRANS_SUM_OUTPUT"]
llo_powerup_data_O4a['9PRG'] = 0.0415 * llo_powerup_data_O4a['L1:LSC-POPAIR_B_RF18_I_MON'] / llo_powerup_data_O4a['L1:IMC-IM4_TRANS_SUM_OUTPUT']
llo_powerup_data_O4a['Parm'] = llo_powerup_data_O4a['L1:IMC-IM4_TRANS_SUM_OUTPUT'] * llo_powerup_data_O4a['PRG'] / 2 * 265

fig, axs = plt.subplots(3,1, sharex = True,figsize =( 6, 12))
lim1 = (250, 350)
lim2= (30, 42)
lim3= (75, 130)
ylims = (lim1, lim2, lim3)
ylabs = ("Arm power [kW]", "carrier PRG", "9 MHz PRG")
for ax, _Pb, _Pa, ylim, _ylab in zip(axs, (llo_powerup_data_O4b["Parm"]/1e3, 
                        llo_powerup_data_O4b["PRG"],
                        llo_powerup_data_O4b["9PRG"]),
                        (llo_powerup_data_O4a["Parm"]/1e3, 
                        llo_powerup_data_O4a["PRG"],
                        llo_powerup_data_O4a["9PRG"]), ylims, ylabs
                        ):
    ax.plot((llo_powerup_data_O4b["time"]) / 3600, _Pb, label="O4b")
    ax.plot((llo_powerup_data_O4b["time"]-720)  / 3600, _Pa, label="O4a")
    ax.set_xscale('log')
    ax.set_xlim(360/ 3600, 10)
    ax.set_ylim(ylim)
    ax.set_ylabel(_ylab)
    ax.legend(loc=3)
ax.set_xlabel("Time [hrs]")

# %%
actuators = {}
actuators["RH"] = {}
actuators["ACO2"] = {}
for chan in list(llo_powerup_data_O4b.keys()):
    if "TCS" in chan:
        if "RH" in chan:
            actuators["RH"][chan] = llo_powerup_data_O4b[chan]
        elif "CO2" in chan:
            actuators["ACO2"][chan] = llo_powerup_data_O4b[chan]
    else:
        continue
    
# %%
# Plotting actuators:
fig, axs = plt.subplots(2, 1, sharex = True,  figsize=(6,12))
for k in list(actuators["RH"].keys()):
    TM = (k.split('-')[1]).split("_")[0]
    axs[0].plot((llo_powerup_data_O4b["time"]) / 3600, 
                actuators["RH"][k], label=TM)
    
for k in list(actuators["ACO2"].keys()):
    TM = (k.split('-')[1]).split("_")[0]
    axs[1].plot((llo_powerup_data_O4b["time"]) / 3600, 
                actuators["ACO2"][k], label=TM)
axs[0].legend(loc=1)
axs[1].legend(loc=4)
axs[0].set_xscale("linear")
axs[0].set_xlim(0, 2)

axs[0].set_ylabel("RH power [W]")
axs[1].set_ylabel("CO2 power [W]")
axs[1].set_xlabel("Time [hr]")
# %%
#  OMC DCPD PSD:
llo_o4b_psd_dat = np.load(llo_psd_O4b, allow_pickle=True)
psd_f = llo_o4b_psd_dat['freq'][0]
llo_psd_time = llo_o4b_psd_dat['time']
llo_o4b_psds = llo_o4b_psd_dat['psd']

# %%
# Filtering psd:
center_f = 10.3e3
bw_f = 0.5e3
f_trunc = psd_f[abs(psd_f - center_f) <= bw_f]
llo_o4b_filt_psds = []
llo_o4b_trunc_psds = []
for _psd in llo_o4b_psds:
    _psd_trunc = _psd[abs(psd_f - center_f) <= bw_f]
    _psd_trunc_medfilt = medfilt(np.log10(_psd_trunc), 81)
    tck = splrep(f_trunc, (_psd_trunc_medfilt),  s=21)
    rec_psd =  10 ** (BSpline(*tck)(f_trunc))
    llo_o4b_filt_psds.append(rec_psd)
    llo_o4b_trunc_psds.append(_psd_trunc)
# %%
def plot_psds(index, psd_time, psds, psds_filtered):
    fig, axs = plt.subplots(2,1, figsize=(6, 12))
    
    # ax0, power:
    axs[0].plot((llo_powerup_data_O4b["time"]) / 3600, llo_powerup_data_O4b['Parm'] / 1e3 )
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_xlabel('time [hr]')
    axs[0].set_ylabel('Arm power [kW]')
    
    it_psd = psd_time[index]
    t_closest = llo_powerup_data_O4b["time"][np.argmin(abs(llo_powerup_data_O4b["time"] - it_psd))] / 3600
    ylim0 = axs[0].get_ylim()
    tline0, = axs[0].plot((t_closest, t_closest), ylim0, 'gray', lw=1)
    axs[0].set_xlim(60/3600, 10)
    # ax1, psd
    psd_trc = {}
    _psd = psds[index]
    _fpsd = psds_filtered[index]

    psd_trc["raw"], = axs[1].plot(f_trunc / 1e3, _psd, 'C2', alpha=0.3)
    psd_trc["filtered"], = axs[1].plot(f_trunc  / 1e3, _fpsd, 'C2')
    axs[1].set_yscale("log")
    axs[1].set_xlabel('frequency [kHz]')
    axs[1].set_ylabel('PSD [a.u.]')
    axs[1].set_ylim(1e-2, 1e3)

    return fig, axs, tline0, psd_trc
# %%
fig, axs, tline0, psd_traces = plot_psds(-1, llo_psd_time, llo_o4b_trunc_psds, llo_o4b_filt_psds)
y0lim = axs[0].get_ylim()
def update_psd(index, freq, psds_raw, psds_filtered, tline, psd_traces):
    it = llo_psd_time[index] / 3600
    tline.set_data((it, it), y0lim)

    psd_traces["raw"].set_data(freq, psds_raw[index])
    psd_traces["filtered"].set_data(freq, psds_filtered[index])
    return tline, psd_traces

ani = animation.FuncAnimation(fig, update_psd, len(llo_psd_time), interval=100,
                              fargs = [f_trunc/1e3, llo_o4b_trunc_psds, llo_o4b_filt_psds,
                                       tline0, psd_traces ])
writervideo = animation.FFMpegWriter(fps=5) 
ani.save('./figures/o4b_omc_dcpd_powerup.mp4',writer=writervideo) 
# %%
