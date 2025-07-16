# %%
import matplotlib.pyplot as plt
import finesse
import finesse.analysis.actions as fac
import finesse.components
import finesse_ligo
import numpy as np
from scipy.optimize import curve_fit
from finesse_ligo.factory import ALIGOFactory
from finesse_ligo.actions import InitialLockLIGO, DARM_RF_to_DC

finesse.init_plotting()

# %% Load in the parameter file
factory = ALIGOFactory("lho_O4.yaml")

# Can also use a free mass here if you want
# from finesse.components.mechanical import FreeMass
# mass = 40
# factory.options.QUAD_suspension_model = FreeMass
# factory.options.QUAD_suspension_kwargs = {"mass": mass}

factory.options.QUAD_suspension_model = finesse_ligo.suspension.QUADSuspension
factory.options.ASC.add = True
factory.options.ASC.close_AC_loops = False 
mass = 40  # mass is baked into the QUAD state-space model
lho = factory.make()
lho.fsig.f = 1

lho.parse(
    """
    readout_dc TRX ETMX.p2.o output_detectors=True
    readout_dc TRY ETMY.p2.o output_detectors=True
    free_mass SRM_sus SRM.mech mass=1
"""
)
# %%
lho.modes(maxtem=4)
lock = InitialLockLIGO(
    exception_on_lock_fail=False,
    lock_steps=100,
    gain_scale=0.4,
    pseudo_lock_arms=False,
    run_locks=True,
)
sol = lho.run(lock)
lho.run(DARM_RF_to_DC())
# %%
def SRCL_dither(model, f, f_min):
    sol = model.run(
        fac.Series(
            # Measure transfer functions from SRCL control signal to
            # arm transmission and the induced DARM motions
            fac.FrequencyResponse(
                f,
                "SRM.mech.F_z",
                [
                    "TRX.DC",
                    "TRY.DC",
                    "DARM.AC.o",
                ],
                name="fresp",
            ),
            fac.DCFields(name="dc"),
        )
    )
    dc = sol["dc"]
    fresp = sol["fresp"]

    P_TRX = np.sum(abs(dc["ETMX.p2.o"]) ** 2, -1).squeeze()[0]
    P_TRY = np.sum(abs(dc["ETMY.p2.o"]) ** 2, -1).squeeze()[0]

    # Normalise into RIN by dividing by DC power
    TRX_SRCL = fresp["TRX.DC", "SRM.mech.F_z"] / P_TRX
    TRY_SRCL = fresp["TRY.DC", "SRM.mech.F_z"] / P_TRY
    DARM_SRCL = fresp["DARM.AC.o", "SRM.mech.F_z"]

    # calculate DARM/TRX,Y transfer functions
    DARM_TRX = DARM_SRCL / TRX_SRCL
    DARM_TRY = DARM_SRCL / TRY_SRCL

    # choose the minimum frequency to fit
    f_slice = f >= f_min
    _f = f[f_slice]


    # fit the transfer functions to 1/f**2 or 1/f**4 behavior

    popt, _ = curve_fit(lambda x, a: a / x**4, _f, abs(DARM_SRCL)[f_slice])
    alpha_1 = popt[0]

    popt, _ = curve_fit(lambda x, a: a / x**2, _f, abs(TRX_SRCL)[f_slice])
    alpha_2 = popt[0]

    popt, _ = curve_fit(lambda x, a: a / x**2, _f, abs(TRY_SRCL)[f_slice])
    alpha_3 = popt[0]

    P_Xarm_est = alpha_1 / alpha_2 * np.pi**2 * mass * finesse.constants.C_LIGHT
    P_Yarm_est = alpha_1 / alpha_3 * np.pi**2 * mass * finesse.constants.C_LIGHT

    P_arm_est = (P_Xarm_est + P_Yarm_est)/2

    # get actual arm power from model
    P_arm_act = (abs(dc["ETMX.p1.i"][0, 0]) ** 2).sum()
    

    plt.figure()
    plt.loglog(fresp.f, abs(TRX_SRCL), label="TRX/SRCL")
    plt.loglog(fresp.f, abs(TRY_SRCL), label="TRY/SRCL")

    plt.loglog(fresp.f, alpha_2 / fresp.f**2, ls="--", label=r"$\alpha_2 / f^{2}$")
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude, [RIN/N]")
    plt.xlim(10, 100)
    plt.ylim(1e-1, 50)
    plt.title(f'TR/SRCL\nDHARD_P offset ={model.DHARD_P.DC/1e-9:.0f} nrad, SRM_P offset = {model.SRM.ybeta/1e-9:.0f}nrad')

    plt.figure()
    plt.loglog(fresp.f, abs(DARM_SRCL), label="DARM/SRCL")
    plt.loglog(fresp.f, alpha_1 / fresp.f**4, ls="--", label=r"$\alpha_1 / f^{4}$")
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude, [m/N]")
    plt.xlim(10, 100)
    plt.ylim(1e-10, 1e-6)
    plt.title(f'DARM/SRCL\nDHARD_P offset ={model.DHARD_P.DC/1e-9:.0f} nrad, SRM_P offset = {model.SRM.ybeta/1e-9:.0f}nrad')


    

    return P_arm_est, P_arm_act
# %%
f = np.geomspace(10, 100, 1000)

with lho.temporary_parameters():


    lho.run("run_locks()")
    P_arm_est, P_arm_act = SRCL_dither(lho, f, 20)
    print(f'Estimated arm power {P_arm_est/1e3} kW, Actual arm power {P_arm_act/1e3} kW')

# %%
f = np.geomspace(10, 100, 1000)

for lho.DHARD_P.DC in np.linspace(-3e-9, 3e-9, 7):
    with lho.temporary_parameters():
        # misalign SRM in pitch by 1 nanorad
        lho.SRM.ybeta = 100e-9
        # misalign DHARD pitch by 1 nanorad
        lho.DHARD_P.DC = 3e-9

        #lho.ETMX.Rc = 2260
        #lho.ETMY.Rc = 2235

        lho.run("run_locks()")
        P_arm_est, P_arm_act = SRCL_dither(lho, f, 20)
        print(f'Estimated arm power {P_arm_est/1e3} kW, Actual arm power {P_arm_act/1e3} kW')



# %%
# detune SRCL
detune_SRCL = np.array([-10, -5, 0, 5, 10])

with lho.temporary_parameters():
    plt.figure()
    for offset in detune_SRCL:
        lho.SRCL.DC = lho.SRCL.DC + offset
        lho.run("run_locks()")
        sol = lho.run(
        fac.Series(
            # Measure transfer functions from SRCL control signal to
            # arm transmission and the induced DARM motions
            fac.FrequencyResponse(
                f,
                "SRM.mech.F_z",
                [
                    "TRX.DC",
                    "TRY.DC",
                    "DARM.AC.o",
                ],
                name="fresp",
            ),
            fac.DCFields(name="dc"),
        )
        )
        dc = sol["dc"]
        fresp = sol["fresp"]

        P_TRX = np.sum(abs(dc["ETMX.p2.o"]) ** 2, -1).squeeze()[0]
        P_TRY = np.sum(abs(dc["ETMY.p2.o"]) ** 2, -1).squeeze()[0]

        # Normalise into RIN by dividing by DC power
        TRX_SRCL = fresp["TRX.DC", "SRM.mech.F_z"] / P_TRX
        TRY_SRCL = fresp["TRY.DC", "SRM.mech.F_z"] / P_TRY
        DARM_SRCL = fresp["DARM.AC.o", "SRM.mech.F_z"]

        # calculate DARM/TRX,Y transfer functions
        DARM_TRX = DARM_SRCL / TRX_SRCL
        DARM_TRY = DARM_SRCL / TRY_SRCL

        
        plt.loglog(fresp.f, abs(DARM_TRX), label=f"DARM/TRX [m/RIN], SRC detuning = {offset} deg")
        plt.loglog(fresp.f, abs(DARM_TRY), label=f"DARM/TRY [m/RIN], SRC detuning = {offset} deg")
    plt.ylim(1e-10, 1e-7)
    #plt.loglog(fresp.f, alpha_X / fresp.f**2, ls="--", label=r"$\alpha_X / f^{2}$")
    plt.loglog()
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.xlim(10, max(f))

# %%
# add change in test mass curvatures

with lho.temporary_parameters():
    plt.figure()
    lho.run("run_locks()")

    sol = lho.run(
    fac.Series(
        # Measure transfer functions from SRCL control signal to
        # arm transmission and the induced DARM motions
        fac.FrequencyResponse(
            f,
            "SRM.mech.F_z",
            [
                "TRX.DC",
                "TRY.DC",
                "DARM.AC.o",
            ],
            name="fresp",
        ),
        fac.DCFields(name="dc"),
    )
    )
    dc = sol["dc"]
    fresp = sol["fresp"]

    P_TRX = np.sum(abs(dc["ETMX.p2.o"]) ** 2, -1).squeeze()[0]
    P_TRY = np.sum(abs(dc["ETMY.p2.o"]) ** 2, -1).squeeze()[0]

    # Normalise into RIN by dividing by DC power
    TRX_SRCL = fresp["TRX.DC", "SRM.mech.F_z"] / P_TRX
    TRY_SRCL = fresp["TRY.DC", "SRM.mech.F_z"] / P_TRY
    DARM_SRCL = fresp["DARM.AC.o", "SRM.mech.F_z"]

    # calculate DARM/TRX,Y transfer functions
    DARM_TRX = DARM_SRCL / TRX_SRCL
    DARM_TRY = DARM_SRCL / TRY_SRCL

    plt.loglog(fresp.f, abs(DARM_TRX), label=f"DARM/TRX [m/RIN], nominal RoC")
    plt.loglog(fresp.f, abs(DARM_TRX), label=f"DARM/TRY [m/RIN], nominal RoC")

    lho.ITMX.Rc = 1950.77
    lho.ITMY.Rc = 1940.3
    lho.ETMX.Rc = 2254.35
    lho.ETMY.Rc = 2250.35
    
    lho.run("run_locks()")
    sol = lho.run(
    fac.Series(
        # Measure transfer functions from SRCL control signal to
        # arm transmission and the induced DARM motions
        fac.FrequencyResponse(
            f,
            "SRM.mech.F_z",
            [
                "TRX.DC",
                "TRY.DC",
                "DARM.AC.o",
            ],
            name="fresp",
        ),
        fac.DCFields(name="dc"),
    )
    )
    dc = sol["dc"]
    fresp = sol["fresp"]

    P_TRX = np.sum(abs(dc["ETMX.p2.o"]) ** 2, -1).squeeze()[0]
    P_TRY = np.sum(abs(dc["ETMY.p2.o"]) ** 2, -1).squeeze()[0]

    # Normalise into RIN by dividing by DC power
    TRX_SRCL = fresp["TRX.DC", "SRM.mech.F_z"] / P_TRX
    TRY_SRCL = fresp["TRY.DC", "SRM.mech.F_z"] / P_TRY
    DARM_SRCL = fresp["DARM.AC.o", "SRM.mech.F_z"]

    # calculate DARM/TRX,Y transfer functions
    DARM_TRX = DARM_SRCL / TRX_SRCL
    DARM_TRY = DARM_SRCL / TRY_SRCL

    plt.loglog(fresp.f, abs(DARM_TRX), label=f"DARM/TRX [m/RIN], change in TM curvature")
    plt.loglog(fresp.f, abs(DARM_TRY), label=f"DARM/TRY [m/RIN], change in TM curvature")
    plt.ylim(1e-10, 1e-7)
    #plt.loglog(fresp.f, alpha_X / fresp.f**2, ls="--", label=r"$\alpha_X / f^{2}$")
    plt.loglog()
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.xlim(10, max(f))
# %%
