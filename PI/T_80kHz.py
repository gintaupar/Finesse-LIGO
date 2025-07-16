import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scipy.constants as scc
import subprocess
import os
import pytest
from fnmatch import filter

from munch import Munch
import finesse.analysis.actions as fa
from finesse.ligo.factory import aligo
from finesse.knm import Map
from finesse.plotting import plot_field
from finesse.ligo.maps import aligo_O4_TM_aperture
from wield.control.plotting import plotTF

from mechanical_modes import MechanicalMode
from optical_modes import CavityModes
from thermal_factory import ALIGOThermalFactory


def ax_to_kHz(ax, scale="linear"):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:0.1f}".format(x * 1e-3)))
    ax.set_xscale(scale)


def savefig(ax, post="", ptype="pdf"):
    fig = ax.figure
    fig.savefig(f"{fig._fname}{post}.{ptype}")


def T_eigenmodes(tpath_join, ppath_join, fpath_join, makegrid):
    # factory = aligo.ALIGOFactory(ppath_join("LHO", "lho_O4.yaml"))
    factory = ALIGOThermalFactory(ppath_join("LHO", "lho_O4.yaml"))
    factory.update_parameters(ppath_join("LHO", "lho_mcmc_RC_lengths.yaml"))
    factory.options.apertures.add = True
    factory.options.apertures.use_surface_profile = True
    factory.options.thermal.add = True
    factory.options.thermal.CO2 = "annular"
    factory.params.CO2 = Munch(
        P_CO2X=1.7,
        P_CO2Y=1.7,
        CO2_data=fpath_join("CO2_Annular_Projection_I_1W.txt"),
    )
    model = factory.make()
    model.P_XARM = 370e3
    model.P_YARM = 370e3
    model.P_RH_ITMX = 0.44
    model.P_RH_ETMX = 1
    model.P_RH_ITMY = 0
    model.P_RH_ETMY = 1
    model.modes(maxtem=1)
    solX = model.run(fa.Eigenmodes("cavXARM", 0))
    solY = model.run(fa.Eigenmodes("cavYARM", 0))

    loss = lambda x: (1 - np.abs(x.eigvalues)**2)
    gouy_deg = lambda x: np.angle(sol.eigvalues) * 180 / np.pi

    fig, ax = plt.subplots()
    ax.semilogy(loss(solX), "o", label="XARM")
    ax.semilogy(loss(solY), "P", label="YARM")
    makegrid(ax)
    ax.legend(loc="upper left")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Roundtrip loss")
    fig.savefig(tpath_join("loss.pdf"))

    fnamesX = []
    fnamesY = []
    qX = model.cavXARM.source.q
    qY = model.cavYARM.source.q
    x_m = np.linspace(-0.17, 0.17, 100) - 14.3e-3
    y_m = np.linspace(-0.17, 0.17, 100) + 16.2e-3
    for idx in range(len(model.homs)):
        figX, axX = plt.subplots()
        figY, axY = plt.subplots()
        plot_field(model.homs, solX.eigvectors[:, idx], qX, ax=axX, x=x_m, y=y_m)
        plot_field(model.homs, solY.eigvectors[:, idx], qY, ax=axY, x=x_m, y=y_m)
        axX.set_title(f"XARM mode {idx}, loss {loss(solX)[idx]}")
        axY.set_title(f"YARM mode {idx}, loss {loss(solY)[idx]}")
        fnameX = tpath_join(f"Xprofile{idx}.pdf")
        fnameY = tpath_join(f"Yprofile{idx}.pdf")
        fnamesX.append(fnameX)
        fnamesY.append(fnameY)
        figX.savefig(fnameX)
        figY.savefig(fnameY)
        for fig in [figX, figY]:
            plt.close(fig)

    cmdX = ["pdftk"] + fnamesX + ["output", tpath_join("XARM_beamshapes.pdf")]
    cmdY = ["pdftk"] + fnamesY + ["output", tpath_join("YARM_beamshapes.pdf")]
    subprocess.call(cmdX)
    subprocess.call(cmdY)
    for fname in fnamesX + fnamesY:
        os.remove(fname)


pstyles = Munch()
lw = 2.8
pstyles.lho = Munch(
    X01=dict(c="xkcd:cerulean", label="X01", ls="-", lw=lw),
    X10=dict(c="xkcd:strawberry", label="X10", ls="-", lw=lw),
    Y01=dict(c="xkcd:goldenrod", label="Y01", ls="--", lw=lw),
    Y10=dict(c="xkcd:kelly green", label="Y10", ls="--", lw=lw),
)
pstyles.llo = Munch(
    X01=dict(c="xkcd:true blue", label="X01", ls="-", lw=lw),
    X10=dict(c="xkcd:tangerine", label="X10", ls="--", lw=lw),
    Y01=dict(c="xkcd:green", label="Y01", ls="-", lw=lw),
    Y10=dict(c="xkcd:fuchsia", label="Y10", ls="--", lw=lw),
)

RH_POWER=Munch(
    lho=Munch(
        P_RH_ETMX=1,
        P_RH_ITMX=0.44,
        P_RH_ETMY=1,
        P_RH_ITMY=0,
    ),
    llo=Munch(
        P_RH_ETMX=1.1,
        P_RH_ITMX=0.5,
        P_RH_ETMY=1.2,
        P_RH_ITMY=0.4,
    ),
)
ACO2_POWER=Munch(
    lho=Munch(P_CO2X=1.7, P_CO2Y=1.7),
    llo=Munch(P_CO2X=0, P_CO2Y=0),
)
THERMAL_STATE=Munch(
    RH_POWER=RH_POWER,
    ACO2_POWER=ACO2_POWER,
    ARM_POWER=Munch(lho=375e3, llo=320e3),
)


@pytest.mark.parametrize("ifo", ["lho", "llo"])
@pytest.mark.parametrize("use_RH", [True, False])
@pytest.mark.parametrize("use_surface_profile", [True, False])
# @pytest.mark.parametrize("CO2_type", [None, "annular"])
def T_80kHz_optical_gains(
        ifo, use_RH, use_surface_profile, tpath_join, ppath_join, fpath_join,
        makegrid, pprint,
):
    CO2_type = None
    factory = ALIGOThermalFactory(ppath_join(ifo.upper(), f"{ifo}_O4.yaml"))
    if ifo == "lho":
        factory.update_parameters(
            ppath_join("LHO", "lho_mcmc_RC_lengths.yaml")
        )
    factory.options.apertures.add = True
    factory.options.apertures.use_surface_profile = use_surface_profile
    factory.options.thermal.add = True
    factory.options.thermal.CO2 = CO2_type  # "annular"
    factory.params.CO2 = THERMAL_STATE.ACO2_POWER[ifo]
    factory.params.CO2.CO2_data = fpath_join("CO2_Annular_Projection_I_1W.txt")
    # use_RH = True
    model = factory.make()
    if factory.options.thermal.add:
        for k, RH_POWER in THERMAL_STATE.RH_POWER[ifo].items():
            RH = model.get(k)
            RH.value = RH_POWER * use_RH
    Parm_W = THERMAL_STATE.ARM_POWER[ifo]
    model.P_XARM = Parm_W
    model.P_YARM = Parm_W
    ptitle = ""
    if not use_RH:
        ptitle += "No RH, "
    # if ifo == "lho" and CO2_type is None:
    #     ptitle += "No CO2, "
    if not use_surface_profile:
        ptitle += "No surface maps, "

    model.modes(maxtem=1)
    fsig = model.fsig.f.ref
    F_Hz = np.linspace(80e3, 81e3, 300)
    sol = model.run(
        fa.Series(
            aligo.InitialLock(),
            aligo.DARM_RF_to_DC(),
            fa.FrequencyResponse3(
                F_Hz,
                [
                    (model.ETMX.p1.o, +fsig),
                    (model.ETMX.p1.o, -fsig),
                    (model.ETMY.p1.o, +fsig),
                    (model.ETMY.p1.o, -fsig),
                ],
                [
                    (model.ETMX.p1.o, +fsig),
                    (model.ETMX.p1.o, -fsig),
                    (model.ETMY.p1.o, +fsig),
                    (model.ETMY.p1.o, -fsig),
                ],
                name="gain",
            ),
            fa.Eigenmodes("cavXARM", 0, name="XARM"),
            fa.Eigenmodes("cavYARM", 0, name="YARM"),
        )
    )

    gains = sol["gain"].out
    plot_modes = False

    ##################################################
    # Optical gains
    ##################################################

    fig = plotTF(F_Hz, gains[..., 1, 1, 1, 1], **pstyles.X01)
    plotTF(F_Hz, gains[..., 1, 1, 2, 2], *fig.axes, **pstyles.X10)
    plotTF(F_Hz, gains[..., 3, 3, 1, 1], *fig.axes, **pstyles.Y01)
    plotTF(F_Hz, gains[..., 3, 3, 2, 2], *fig.axes, **pstyles.Y10)
    fig.axes[0].legend()
    fig.axes[1].set_xscale("linear")
    fig.axes[0].set_title(f"{ifo.upper()} Optical gains {ptitle}")
    fig.savefig(tpath_join("gains.pdf"))

    ##################################################
    # PI gains
    ##################################################

    def transfer(arm, hom_idx):
        # arm = 0 for X arm, arm = 1 for Y arm
        aidx = 2 * arm
        Hp = gains[..., aidx, aidx, hom_idx, hom_idx]
        Hl = gains[..., aidx + 1, aidx + 1, hom_idx, hom_idx]
        return Hp - Hl

    M_kg = 40
    cc = 8 * np.pi / (model.lambda0 * scc.c * M_kg * (2 * np.pi)**2) * Parm_W

    def normalized_gain(arm, hom_idx):
        Hsb = transfer(arm, hom_idx)
        return -cc / F_Hz**2 * np.real(Hsb)

    pi_gains = Munch(
        X01=normalized_gain(0, 1),
        X10=normalized_gain(0, 2),
        Y01=normalized_gain(1, 1),
        Y10=normalized_gain(1, 2),
    )

    fig_amp, ax_amp = plt.subplots()
    fig_sgn, ax_sgn = plt.subplots()
    fig_crt, ax_crt = plt.subplots()
    for k in ["X01", "X10", "Y01", "Y10"]:
        ax_amp.semilogy(F_Hz, np.abs(pi_gains[k]), **pstyles[k])
        ax_sgn.plot(F_Hz, np.sign(pi_gains[k]), **pstyles[k])
        ax_crt.semilogy(F_Hz, 1 / np.abs(pi_gains[k]), **pstyles[k])
    for ax in [ax_amp, ax_sgn, ax_crt]:
        ax.legend()
        makegrid(ax, F_Hz)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:0.1f}".format(x*1e-3)))
        ax.set_xlabel("Frequency [kHz]")
    ax_amp.set_title(f"{ifo.upper()} Normalized PI gain amplitude {ptitle}")
    ax_sgn.set_title(f"{ifo.upper()} PI gain sign {ptitle}")
    ax_crt.set_title(f"{ifo.upper()} Critical Q overlap product {ptitle}")
    fig_amp.savefig(tpath_join("pi_gains.pdf"))
    fig_sgn.savefig(tpath_join("pi_signs.pdf"))
    fig_crt.savefig(tpath_join("critical_product.pdf"))


    if plot_modes:
        loss_x = 1 - np.abs(sol["XARM"].eigvalues)**2
        loss_y = 1 - np.abs(sol["YARM"].eigvalues)**2
        gouy_x = np.angle(sol["XARM"].eigvalues, deg=True)
        gouy_y = np.angle(sol["YARM"].eigvalues, deg=True)

        fig, ax = plt.subplots()
        ax.plot(loss_x * 100, "o", label="XARM")
        ax.plot(loss_y * 100, "^", label="YARM")
        makegrid(ax)
        ax.legend()
        ax.set_ylabel("Loss [percent]")
        fig.savefig(tpath_join("loss.pdf"))

        fig, ax = plt.subplots()
        ax.plot(gouy_x, "o", label="XARM")
        ax.plot(gouy_y, "^", label="YARM")
        makegrid(ax)
        ax.legend()
        ax.set_ylabel("Gouy phase [deg]")
        fig.savefig(tpath_join("gouy.pdf"))

        def save_profile(esol, idx, fname):
            fig, ax = plt.subplots()
            plot_field(
                model.homs, esol.eigvectors[:, idx], model.cavXARM.source.q, ax=ax,
            )
            ax.set_title("loss {:0.3f}%".format((1 - np.abs(esol.eigvalues[idx])**2) * 100))
            fig.savefig(fname)
            # return fig

        save_profile(sol["XARM"], 1, tpath_join("Xprofile01.pdf"))
        save_profile(sol["XARM"], 2, tpath_join("Xprofile10.pdf"))
        save_profile(sol["YARM"], 1, tpath_join("Yprofile01.pdf"))
        save_profile(sol["YARM"], 2, tpath_join("Yprofile10.pdf"))


@pytest.mark.parametrize("ifo", ["lho", "llo"])
def T_single_parameters(
        ifo, tpath_join, fpath_join, ppath_join, makegrid, pprint):
    # ifo = "llo"
    factory = ALIGOThermalFactory(ppath_join(ifo.upper(), f"{ifo}_O4.yaml"))
    if ifo == "lho":
        factory.update_parameters(
            ppath_join("LHO", "lho_mcmc_RC_lengths.yaml")
        )
    factory.options.apertures.add = True
    factory.options.apertures.use_surface_profile = True
    factory.options.thermal.add = True
    factory.options.thermal.CO2 = None  # "annular"
    factory.params.CO2 = THERMAL_STATE.ACO2_POWER[ifo]
    factory.params.CO2.CO2_data = fpath_join("CO2_Annular_Projection_I_1W.txt")
    use_RH = True
    model = factory.make()
    if factory.options.thermal.add:
        for k, RH_POWER in THERMAL_STATE.RH_POWER[ifo].items():
            RH = model.get(k)
            RH.value = RH_POWER * use_RH
    Parm_W = THERMAL_STATE.ARM_POWER[ifo]
    model.P_XARM = Parm_W
    model.P_YARM = Parm_W
    # model.ls1.L.value += 0.02
    # model.SR2.Rc -= 0.002

    model.modes(maxtem=1)
    fsig = model.fsig.f.ref

    if ifo == "lho":
        F_Hz = np.linspace(80.3e3, 80.6e3, 600)
    elif ifo == "llo":
        F_Hz = np.linspace(79.9e3, 80.6e3, 600)
    sol = model.run(
        fa.Series(
            aligo.InitialLock(),
            aligo.DARM_RF_to_DC(),
            fa.FrequencyResponse3(
                F_Hz,
                [
                    (model.ETMX.p1.o, +fsig),
                    (model.ETMX.p1.o, -fsig),
                    (model.ETMY.p1.o, +fsig),
                    (model.ETMY.p1.o, -fsig),
                ],
                [
                    (model.ETMX.p1.o, +fsig),
                    (model.ETMX.p1.o, -fsig),
                    (model.ETMY.p1.o, +fsig),
                    (model.ETMY.p1.o, -fsig),
                ],
                name="PI_gains",
            ),
            fa.FrequencyResponse2(
                F_Hz,
                [
                    (model.ETMX.p1.o, +fsig),
                    (model.ETMX.p1.o, -fsig),
                    (model.ETMY.p1.o, +fsig),
                    (model.ETMY.p1.o, -fsig),
                ],
                [model.AS.DC.o],
                name="AS_PI",
            ),
        )
    )

    check_DARM = False
    if check_DARM:
        Fd_Hz = np.geomspace(10, 10e3, 200)
        sol_DARM = model.run(fa.FrequencyResponse(Fd_Hz, [model.DARM.AC], [model.AS.DC], name="DARM"))
        sol_DARM = model.run(
            fa.Series(
                aligo.InitialLock(),
                aligo.DARM_RF_to_DC(),
                fa.FrequencyResponse(Fd_Hz, [model.DARM.AC], [model.AS.DC], name="DARM"),
            ),
        )
        fig = plotTF(Fd_Hz, sol_DARM["DARM"].out[..., 0, 0])
        # fig = plotTF(Fd_Hz, sol_DARM.out[..., 0, 0])
        fig.savefig(tpath_join("DARM.pdf"))

    gains = sol["PI_gains"].out
    aidx = dict(X=0, Y=2)
    M_kg = 40
    cc = 8 * np.pi / (model.lambda0 * scc.c * M_kg * (2 * np.pi)**2) * Parm_W


    def normalized_gain(arm, hom_idx):
        idx = aidx[arm]
        Hp = sol["PI_gains"].out[..., idx, idx, hom_idx, hom_idx]
        Hl = sol["PI_gains"].out[..., idx + 1, idx + 1, hom_idx, hom_idx]
        Hsb = Hp - Hl
        return -cc / F_Hz**2 * np.real(Hsb)

    pi_gains = Munch(
        X01=normalized_gain("X", 1),
        X10=normalized_gain("X", 2),
        Y01=normalized_gain("Y", 1),
        Y10=normalized_gain("Y", 2),
    )
    fmax = Munch({k: F_Hz[np.argmax(np.abs(gain))] for k, gain in pi_gains.items()})

    fig_amp, ax_amp = plt.subplots()
    fig_crt, ax_crt = plt.subplots()
    plot_tfs = False
    for idx, k in enumerate(["X01", "X10", "Y01", "Y10"]):
        print(f"{k} all positive gains? {np.all(np.sign(np.abs(pi_gains[k])) == 1)}")
        ax_amp.semilogy(F_Hz, np.abs(pi_gains[k]), **pstyles[ifo][k])
        ax_crt.semilogy(F_Hz, 1 / np.abs(pi_gains[k]), **pstyles[ifo][k])
        pidx = aidx[k[0]] + 1
        hom_idx = int(k[1]) + 1
        Hl_opt = sol["PI_gains"].out[..., pidx, pidx, hom_idx, hom_idx]
        Hl_dc = sol["AS_PI"].out[..., 0, pidx, hom_idx]
        if plot_tfs:
            if idx == 0:
                fig_opt = plotTF(F_Hz, Hl_opt, **pstyles[ifo][k])
                fig_dc = plotTF(F_Hz, Hl_dc, **pstyles[ifo][k])
            else:
                plotTF(F_Hz, Hl_opt, *fig_opt.axes, **pstyles[ifo][k])
                plotTF(F_Hz, Hl_dc, *fig_dc.axes, **pstyles[ifo][k])
    for ax in [ax_amp, ax_crt]:
        makegrid(ax, F_Hz)
        ax.set_xlabel("Frequency [Hz]", fontsize=12)
    gain_axs = [ax_amp, ax_crt]
    ax_amp.set_title(f"{ifo.upper()} Normalized PI gain magnitude")
    ax_crt.set_title(f"{ifo.upper()} Critical Q overlap product")
    fig_amp._fname = tpath_join("pi_gains")
    fig_crt._fname = tpath_join("critical_product")

    ptype = "pdf"
    if plot_tfs:
        # leg_axs.extend([fig_opt.axes[0], fig_dc.axes[0]])
        fig_opt.axes[0].set_title(f"{ifo.upper()} PI optical gains")
        fig_dc.axes[0].set_title(f"{ifo.upper()} PI to AS DC")
        fig_dc.axes[0].set_xscale("linear")
        fig_opt.axes[0].set_xscale("linear")
        fig_opt._fname = tpath_join("optical_gains")
        fig_dc._fname = tpath_join("pi_as")
        tf_phs = [fig_opt.axes[1], fig_dc.axes[1]]
        tf_amp = [fig_opt.axes[0], fig_dc.axes[0]]
    else:
        tf_phs = []
        tf_amp = []
    for ax in gain_axs + tf_amp:
        ax.legend()
        savefig(ax, ptype=ptype)
    for ax in gain_axs + tf_phs:
        if "critical_product" in ax.figure._fname:
            ax.set_ylim(2e5, 2e7)
        f10 = (fmax.X10 + fmax.Y10) / 2
        f01 = (fmax.X01 + fmax.Y01) / 2
        df = 5
        ax.set_xlim(f10 - df, f10 + df)
        xticks = f10 - np.arange(-4, 5, 2)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x - f10:0.1f}"))
        ax.set_xlabel(
            f"Frequency offset [Hz] from max gain at {f10:0.0f} Hz", fontsize=12,
        )
        savefig(ax, "10", ptype=ptype)
        ax.set_xlim(f01 - df, f01 + df)
        xticks = f01 - np.arange(-4, 5, 2)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x - f01:0.1f}"))
        ax.set_xlabel(
            f"Frequency offset [Hz] from max gain at {f01:0.0f} Hz", fontsize=12,
        )
        savefig(ax, "01", ptype=ptype)


@pytest.mark.parametrize("ifo", ["lho", "llo"])
def T_compare_effects(ifo, tpath_join, ppath_join, fpath_join, makegrid, pprint):
    factory = ALIGOThermalFactory(ppath_join(ifo.upper(), f"{ifo}_O4.yaml"))
    if ifo == "lho":
        factory.update_parameters(
            ppath_join("LHO", "lho_mcmc_RC_lengths.yaml")
        )

    # make models
    factory.options.thermal.add = True
    models = Munch()
    models.SH = factory.make()  # self-heating
    print(models.SH.SRCL)
    models.RH = factory.make()  # self + ring heater
    # models.CO2 = factory.make()  # self + ring heater + CO2
    # factory.options.apertures.add = True
    factory.options.apertures.add = True
    factory.options.BS_HR = False
    factory.options.PR3_SR3 = False
    models.AP = factory.make()  # self + RH + apertures
    factory.options.apertures.use_surface_profile = True
    models.SP = factory.make()  # self + RH + apertures + surface maps

    # set up thermal states
    Parm_W = THERMAL_STATE.ARM_POWER[ifo]
    for model in models.values():
        model.P_XARM = Parm_W
        model.P_YARM = Parm_W
        model.modes(maxtem=1)
    for mkey, model in models.items():
        # self heating gets no thermal corrections
        if mkey == "SH":
            continue
        # everyone else has ring heaters
        for k, RH_POWER in THERMAL_STATE.RH_POWER[ifo].items():
            RH = model.get(k)
            RH.value = RH_POWER

    F_Hz = np.linspace(80e3, 81e3, 300)

    def actions(model):
        fsig = model.fsig.f.ref
        PI_PORTS = [
            (model.ETMX.p1.o, +fsig),
            (model.ETMX.p1.o, -fsig),
            (model.ETMY.p1.o, +fsig),
            (model.ETMY.p1.o, -fsig),
        ]
        return [
            aligo.InitialLock(),
            aligo.DARM_RF_to_DC(),
            fa.FrequencyResponse3(F_Hz, PI_PORTS, PI_PORTS, name="PI_gains"),
        ]

    sols = Munch({
        key: model.run(fa.Series(*actions(model))) for key, model in models.items()
    })

    aidx = dict(X=0, Y=2)
    M_kg = 40
    cc = 8 * np.pi / (1064e-9 * scc.c * M_kg * (2 * np.pi)**2) * Parm_W

    def normalized_gain(sol, arm, hom_idx):
        idx = aidx[arm]
        Hp = sol["PI_gains"].out[..., idx, idx, hom_idx, hom_idx]
        Hl = sol["PI_gains"].out[..., idx + 1, idx + 1, hom_idx, hom_idx]
        return Hp - Hl

        Hsb = transfer(arm, hom_idx)
        return -cc / Fpi_Hz**2 * np.real(Hp - Hl)

    # pi_gains = Munch(
    #     X01=normalized_gain("X", 1),
    #     X10=normalized_gain("X", 2),
    #     Y01=normalized_gain("Y", 1),
    #     Y10=normalized_gain("Y", 2),
    # )

    arm = "X"
    hom_idx = 2
    fig_amp, ax_amp = plt.subplots()
    fig_crt, ax_crt = plt.subplots()
    for k, sol in sols.items():
        ng = np.abs(normalized_gain(sol, arm, hom_idx))
        ax_amp.semilogy(F_Hz, ng, label=k)
        ax_crt.semilogy(F_Hz, 1 / ng, label=k)
    for ax in [ax_amp, ax_crt]:
        ax.legend()
        makegrid(ax, F_Hz)
        ax.set_xlabel("Frequency")
    fig_amp.savefig(tpath_join(f"{arm}_{hom_idx}_gains.pdf"))
    fig_crt.savefig(tpath_join(f"{arm}_{hom_idx}_crtprd.pdf"))


@pytest.mark.parametrize("ifo", ["lho", "llo"])
def T_operating_point(ifo, tpath_join, ppath_join, pprint, makegrid):
    factory = ALIGOThermalFactory(ppath_join(ifo.upper(), f"{ifo}_O4.yaml"))
    if ifo == "lho":
        factory.update_parameters(
            ppath_join("LHO", "lho_mcmc_RC_lengths.yaml")
        )
    factory.options.apertures.add = True
    factory.options.apertures.use_surface_profile = True
    factory.options.thermal.add = True
    model = factory.make()
    use_RH = True
    if factory.options.thermal.add:
        for k, RH_POWER in THERMAL_STATE.RH_POWER[ifo].items():
            RH = model.get(k)
            RH.value = RH_POWER * use_RH
    Parm_W = THERMAL_STATE.ARM_POWER[ifo]
    model.P_XARM = Parm_W
    model.P_YARM = Parm_W

    model.modes(maxtem=1)
    fsig = model.fsig.f.ref
    PI_ports = [
        (model.ETMX.p1.o, +fsig),
        (model.ETMX.p1.o, -fsig),
        (model.ETMY.p1.o, +fsig),
        (model.ETMY.p1.o, -fsig),
    ]
    Fsens_Hz = np.geomspace(1, 10e3, 300)
    Fpi_Hz = np.linspace(80e3, 81e3, 300)
    sol = model.run(
        fa.Series(
            aligo.InitialLock(),
            aligo.DARM_RF_to_DC(),
            fa.FrequencyResponse(
                Fsens_Hz, [model.DARM.AC], [model.AS.DC],
                name="DARM",
            ),
            fa.FrequencyResponse2(
                Fpi_Hz, PI_ports, [model.AS.DC.o],
                name="AS_PI",
            ),
            fa.FrequencyResponse3(
                Fpi_Hz, PI_ports, PI_ports,
                name="PI_gains",
            ),
        )
    )

    aidx = dict(X=0, Y=2)

    def transfer(arm, hom_idx):
        idx = aidx[arm]
        Hp = sol["PI_gains"].out[..., idx, idx, hom_idx, hom_idx]
        Hl = sol["PI_gains"].out[..., idx + 1, idx + 1, hom_idx, hom_idx]
        return Hp - Hl

    M_kg = 40
    cc = 8 * np.pi / (model.lambda0 * scc.c * M_kg * (2 * np.pi)**2) * Parm_W


    fig = plotTF(Fsens_Hz, sol["DARM"].out[..., 0, 0])
    fig.axes[0].set_title(f"{ifo.upper()} DARM to AS DC")
    fig.savefig(tpath_join("DARM.pdf"))

    fig_amp, ax_amp = plt.subplots()
    fig_sgn, ax_sgn = plt.subplots()
    fig_crt, ax_crt = plt.subplots()
    for idx, k in enumerate(["X01", "X10", "Y01", "Y10"]):
        ax_amp.semilogy(Fpi_Hz, np.abs(pi_gains[k]), **pstyles[k])
        ax_sgn.plot(Fpi_Hz, np.sign(pi_gains[k]), **pstyles[k])
        ax_crt.semilogy(Fpi_Hz, 1 / np.abs(pi_gains[k]), **pstyles[k])
        pidx = aidx[k[0]] + 1
        hom_idx = int(k[1]) + 1
        Hl_opt = sol["PI_gains"].out[..., pidx, pidx, hom_idx, hom_idx]
        Hl_dc = sol["AS_PI"].out[..., 0, pidx, hom_idx]
        if idx == 0:
            fig_opt = plotTF(Fpi_Hz, Hl_opt, **pstyles[k])
            fig_dc = plotTF(Fpi_Hz, Hl_dc, **pstyles[k])
        else:
            plotTF(Fpi_Hz, Hl_opt, *fig_opt.axes, **pstyles[k])
            plotTF(Fpi_Hz, Hl_dc, *fig_dc.axes, **pstyles[k])
    for ax in [ax_amp, ax_sgn, ax_crt]:
        makegrid(ax, Fpi_Hz)
    for ax in [ax_amp, ax_sgn, ax_crt, fig_opt.axes[1], fig_dc.axes[1]]:
        ax_to_kHz(ax)
        ax.set_xlabel("Frequency [kHz]")
    for ax in [ax_amp, ax_sgn, ax_crt, fig_opt.axes[0], fig_dc.axes[0]]:
        ax.legend()
    ax.set_xlabel("Frequency [kHz]")
    ax_amp.set_title(f"{ifo.upper()} Normalized PI gain magnitude")
    ax_sgn.set_title(f"{ifo.upper()} PI gain sign")
    ax_crt.set_title(f"{ifo.upper()} Critical Q overlap product")
    fig_opt.axes[0].set_title(f"{ifo.upper()} PI optical gains")
    fig_dc.axes[0].set_title(f"{ifo.upper()} PI to AS DC")
    fig_amp.savefig(tpath_join("pi_gains.pdf"))
    fig_sgn.savefig(tpath_join("pi_signs.pdf"))
    fig_crt.savefig(tpath_join("critical_product.pdf"))
    fig_opt.savefig(tpath_join("optical_gains.pdf"))
    fig_dc.savefig(tpath_join("pi_as.pdf"))


def T_mechanical_modes(fpath_join, tpath_join):
    mode_files = sorted(filter(os.listdir(fpath_join("Modes_80kHz")), "*mds"))
    print(mode_files)
    fnames = []
    for mode_file in mode_files:
        print(mode_file)
        # data = np.loadtxt(fpath_join("Modes_80kHz", mode_file))
        # Fm_Hz = data[0, -1]
        # assert np.all(data[:, -1] == Fm_Hz)
        # mode = MechanicalMode(
        #     disp_z_data=data[:, 3],
        #     coords_m=data[:, 1:3],
        #     F_Hz=Fm_Hz,
        # )
        mode = MechanicalMode.from_slawek(fpath_join("Modes_80kHz", mode_file))
        x_m = np.linspace(-0.17, 0.17, 198)
        y_m = np.linspace(-0.17, 0.17, 200)
        mode.update_mesh(x_m, y_m)
        fig = mode.plot_mode()
        fig.gca().set_title(f"{mode_file.split('.')[0]}")
        fname = tpath_join(f"{mode_file.split('.')[0]}.pdf")
        fig.savefig(fname)
        fnames.append(fname)
    cmd = ["pdftk"] + fnames + ["output", tpath_join("mode_shapes.pdf")]
    subprocess.call(cmd)
    for fname in fnames:
        os.remove(fname)


def make_full_model(factory, use_surface_profile, ifo="lho"):
    factory.options.apertures.add = True
    factory.options.apertures.use_surface_profile = use_surface_profile
    factory.options.thermal.add = True
    model = factory.make()
    for k, RH_POWER in THERMAL_STATE.RH_POWER["lho"].items():
        RH = model.get(k)
        RH.value = RH_POWER
    Parm_W = THERMAL_STATE.ARM_POWER["lho"]
    model.P_XARM = Parm_W
    model.P_YARM = Parm_W
    return model


def T_optical_modes(tpath_join, ppath_join, pprint):
    factory = ALIGOThermalFactory(ppath_join("LHO", "lho_O4.yaml"))
    factory.update_parameters(ppath_join("LHO", "lho_mcmc_RC_lengths.yaml"))
    model = make_full_model(factory, use_surface_profile=True)
    model.modes(maxtem=1)
    sol = model.run(fa.Eigenmodes("cavXARM", 0))
    modes = CavityModes(sol, model.homs, model.cavXARM.source.q)
    dpitch_m = -14.3e-3
    dyaw_m = +16.2e-3
    x_m = np.linspace(-0.17, 0.17, 98) - dyaw_m
    y_m = np.linspace(-0.17, 0.17, 100) - dpitch_m
    modes.calc_mode_shapes(x_m, y_m)
    pprint("00")
    pprint(modes.calc_mode_overlaps(0, 0))
    pprint(modes.calc_mode_overlaps(0, 1))
    pprint(modes.calc_mode_overlaps(0, 2))
    pprint("01")
    pprint(modes.calc_mode_overlaps(1, 0))
    pprint(modes.calc_mode_overlaps(1, 1))
    pprint(modes.calc_mode_overlaps(1, 2))
    pprint("10")
    pprint(modes.calc_mode_overlaps(2, 0))
    pprint(modes.calc_mode_overlaps(2, 1))
    pprint(modes.calc_mode_overlaps(2, 2))
    for idx in range(len(modes.homs)):
        fig = modes.plot_mode(idx)
        fig.savefig(tpath_join(f"mode{idx}.pdf"))
    fnames = [tpath_join(f"mode{idx}.pdf") for idx in range(len(modes.homs))]
    cmd = ["pdftk"] + fnames + ["output", tpath_join("mode_shapes.pdf")]
    subprocess.call(cmd)
    for fname in fnames:
        os.remove(fname)


@pytest.mark.parametrize("shift_beam", [0, +1, -1])
@pytest.mark.parametrize("use_surface_profile", [True, False])
def T_overlaps(
        shift_beam, use_surface_profile, fpath_join, tpath_join, ppath_join, pprint, makegrid,
):
    plot_modes = True
    mech_names = []
    opt_names = []
    # optical modes
    factory = ALIGOThermalFactory(ppath_join("LHO", "lho_O4.yaml"))
    factory.update_parameters(ppath_join("LHO", "lho_mcmc_RC_lengths.yaml"))
    pprint("using surface profile?", use_surface_profile)
    model = make_full_model(factory, use_surface_profile=use_surface_profile)
    maxtem = 8
    model.modes(maxtem=maxtem)
    with model.temporary_parameters():
        model.ITMX.set_RTL(T=0, L=0)
        model.ETMX.set_RTL(T=0, L=0)
        model.X_arm_loss = 0
        sol = model.run(fa.Eigenmodes("cavXARM", 0))
    optical_modes = CavityModes(sol, model.homs, maxtem, model.cavXARM.source.q)
    # print(optical_modes._mode_order)
    # shift_beam = True
    dpitch_m = -14.3e-3 * np.abs(shift_beam)
    dyaw_m = +16.2e-3 * shift_beam
    pprint(f"beam shifted in pitch by {dpitch_m * 1e3:0.1f} mm")
    pprint(f"beam shifted in yaw by {dyaw_m * 1e3:0.1f} mm")
    x_m = np.linspace(-0.17, 0.17, 98)  # 98
    y_m = np.linspace(-0.17, 0.17, 100)  # 100
    optical_modes.calc_mode_shapes(x_m, y_m, dx_m=dyaw_m, dy_m=dpitch_m)

    # mechanical modes
    mode_files = sorted(filter(os.listdir(fpath_join("Modes_80kHz")), "*mds"))
    for mode_file in mode_files:
        mechanical_mode = MechanicalMode.from_slawek(fpath_join("Modes_80kHz", mode_file))
        mechanical_mode.update_mesh(x_m, y_m)
        overlaps = np.abs(mechanical_mode.compute_all_mode_overlaps(optical_modes))
        midx = np.argmax(overlaps)
        max_overlap = np.max(overlaps)
        # moder = optical_modes.mode_order[midx]
        # print(max_overlap, morder)
        print(max_overlap)
        pprint(mode_file.split(".")[0], overlaps)
        if plot_modes:
            fig = mechanical_mode.plot_mode()
            fig.gca().set_title(
                f"{mode_file.split('.')[0]}, max overlap: {max_overlap:0.2e}, {midx + 1}")
            fname = tpath_join(f"{mode_file.split('.')[0]}.pdf")
            fig.savefig(fname)
            mech_names.append(fname)

    if plot_modes:
        fig, ax = plt.subplots()
        ax.plot(optical_modes.loss_rt, "o")
        makegrid(ax)
        ax.set_xlabel("Mode number")
        ax.set_ylabel("Roundtrip loss")
        ax.set_yscale("log")
        fig.savefig(tpath_join("loss.pdf"))
        fig, ax = plt.subplots()
        ax.plot(optical_modes.gouy_deg, "o")
        ax.set_xlabel("Mode number")
        ax.set_ylabel("Roundtrip Gouy phase [deg]")
        makegrid(ax)
        fig.savefig(tpath_join("gouy_phase.pdf"))
        for idx in range(optical_modes.Nhoms):
            fig = optical_modes.plot_mode(idx)
            loss = optical_modes.loss_rt[idx] * 1e6
            gouy = optical_modes.gouy_deg[idx]
            fig.gca().set_title(
                f"Round trip loss {loss:0.0f} ppm, Round trip Gouy phase {gouy:0.0f} deg"
            )
            fname = tpath_join(f"opt{idx}.pdf")
            fig.savefig(fname)
            opt_names.append(fname)
        cmd = ["pdftk"] + mech_names + ["output", tpath_join("mechanical_modes.pdf")]
        subprocess.call(cmd)
        cmd = ["pdftk"] + opt_names + ["output", tpath_join("optical_modes.pdf")]
        subprocess.call(cmd)
        for fname in mech_names + opt_names:
            os.remove(fname)



def T_CO2(tpath_join, fpath_join):
    CP = thermal_maps.make_CP_model()
    fname = fpath_join("CO2_Annular_Projection_I_1W.txt")
    P_ACO2 = 1
    I_ACO2, _ = thermal_maps.make_ACO2_intensity(P_ACO2, fname)
    ss_aco2 = AdvancedLIGOTestMass3DSteadyState(CP)
    ss_aco2.temperature.I_HR.interpolate(I_ACO2)
    ss_aco2.solve_temperature()

    x, TM_aperture = aligo_O4_TM_aperture()
    y = x
    ACO2_SUB_1W = thermal_maps.get_opd(x, y, ss_aco2)

    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, ACO2_SUB_1W, rasterized=True)
    ax.set_aspect("equal")
    fig.savefig(tpath_join("ACO2.pdf"))


# @pytest.mark.parametrize("ifo", ["lho", "llo"])
def T_15kHz(
        tpath_join, fpath_join, ppath_join, makegrid, pprint):
    ifo = "lho"
    factory = ALIGOThermalFactory(ppath_join(ifo.upper(), f"{ifo}_O4.yaml"))
    if ifo == "lho":
        factory.update_parameters(
            ppath_join("LHO", "lho_mcmc_RC_lengths.yaml")
        )
    factory.options.apertures.add = True
    factory.options.apertures.use_surface_profile = True
    factory.options.thermal.add = True
    factory.options.thermal.CO2 = None
    use_RH = True
    model = factory.make()
    if factory.options.thermal.add:
        for k, RH_POWER in THERMAL_STATE.RH_POWER[ifo].items():
            RH = model.get(k)
            RH.value = RH_POWER * use_RH
    # Parm_W = THERMAL_STATE.ARM_POWER[ifo]
    Parm_W = 50e3
    model.P_XARM = Parm_W
    model.P_YARM = Parm_W

    model.modes(modes=[[0, 0], [1, 3], [3, 1]])
    fsig = model.fsig.f.ref

    F_Hz = np.linspace(10e3, 20e3, 200)
    sol = model.run(
        fa.Series(
            aligo.InitialLock(),
            aligo.DARM_RF_to_DC(),
            fa.FrequencyResponse3(
                F_Hz,
                [
                    (model.ETMX.p1.o, +fsig),
                    (model.ETMX.p1.o, -fsig),
                    (model.ETMY.p1.o, +fsig),
                    (model.ETMY.p1.o, -fsig),
                ],
                [
                    (model.ETMX.p1.o, +fsig),
                    (model.ETMX.p1.o, -fsig),
                    (model.ETMY.p1.o, +fsig),
                    (model.ETMY.p1.o, -fsig),
                ],
                name="PI_gains",
            ),
        )
    )

    aidx = dict(X=0, Y=2)
    M_kg = 40
    cc = 8 * np.pi / (model.lambda0 * scc.c * M_kg * (2 * np.pi)**2) * Parm_W

    def normalized_gain(arm, hom_idx):
        idx = aidx[arm]
        Hp = sol["PI_gains"].out[..., idx, idx, hom_idx, hom_idx]
        Hl = sol["PI_gains"].out[..., idx + 1, idx + 1, hom_idx, hom_idx]
        Hsb = Hp - Hl
        return -cc / F_Hz**2 * np.real(Hsb)

    pi_gains = Munch(
        X01=normalized_gain("X", 1),
        X10=normalized_gain("X", 2),
        Y01=normalized_gain("Y", 1),
        Y10=normalized_gain("Y", 2),
    )
    fmax = Munch({k: F_Hz[np.argmax(np.abs(gain))] for k, gain in pi_gains.items()})

    fig_amp, ax_amp = plt.subplots()
    fig_crt, ax_crt = plt.subplots()
    plot_tfs = False
    for idx, k in enumerate(["X01", "X10", "Y01", "Y10"]):
        print(f"{k} all positive gains? {np.all(np.sign(np.abs(pi_gains[k])) == 1)}")
        ax_amp.semilogy(F_Hz, np.abs(pi_gains[k]), **pstyles[ifo][k])
        ax_crt.semilogy(F_Hz, 1 / np.abs(pi_gains[k]), **pstyles[ifo][k])
    for ax in [ax_amp, ax_crt]:
        makegrid(ax, F_Hz)
        ax.set_xlabel("Frequency [Hz]", fontsize=12)
    gain_axs = [ax_amp, ax_crt]
    ax_amp.set_title(f"{ifo.upper()} Normalized PI gain magnitude")
    ax_crt.set_title(f"{ifo.upper()} Critical Q overlap product")
    fig_amp._fname = tpath_join("pi_gains")
    fig_crt._fname = tpath_join("critical_product")

    ptype = "pdf"
    for ax in gain_axs:
        ax.legend()
        savefig(ax, ptype=ptype)
