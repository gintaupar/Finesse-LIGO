import numpy as np
import scipy.constants as scc
import matplotlib.pyplot as plt
import os
import pytest

from munch import Munch
import finesse.analysis.actions as fa
from finesse.ligo.factory import aligo
from wield.utilities.file_io import load, save
from wield.control.plotting import plotTF

from thermal_factory import ALIGOThermalFactory


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


@pytest.fixture
def optical_gains(ppath_join, fpath_join):
    force_calc = False
    zoom = True
    ifo = "lho"
    if zoom:
        fname = fpath_join(f"optical_gains_{ifo}_zoom.h5")
        F_Hz = np.linspace(80.35e3, 80.45e3, 600)
    else:
        fname = fpath_join(f"optical_gains_{ifo}.h5")
        F_Hz = np.linspace(80e3, 81e3, 600)

    if os.path.exists(fname) and not force_calc:
        print("loading")
        data = Munch(load(fname))
    else:
        print("calculating")
        print(f"min {F_Hz[0]} Hz, max {F_Hz[-1]} Hz")
        use_RH = True
        CO2_type = None  # "annular"
        use_surface_profile = True
        factory = ALIGOThermalFactory(ppath_join(ifo.upper(), f"{ifo}_O4.yaml"))
        if ifo == "lho":
            factory.update_parameters(
                ppath_join("LHO", "lho_mcmc_RC_lengths.yaml")
            )
        factory.options.apertures.add = True
        factory.options.apertures.use_surface_profile = use_surface_profile
        factory.options.thermal.add = True
        factory.options.thermal.CO2 = CO2_type
        factory.params.CO2 = THERMAL_STATE.ACO2_POWER[ifo]
        factory.params.CO2.CO2_data = fpath_join("CO2_Annular_Projection_I_1W.txt")

        model = factory.make()
        if factory.options.thermal.add:
            for k, RH_POWER in THERMAL_STATE.RH_POWER[ifo].items():
                RH = model.get(k)
                RH.value = RH_POWER * use_RH
        Parm_W = THERMAL_STATE.ARM_POWER[ifo]
        model.P_XARM = Parm_W
        model.P_YARM = Parm_W

        model.modes(maxtem=10)
        fsig = model.fsig.f.ref

        sol = model.run(
            fa.Series(
                # aligo.InitialLock(),
                # aligo.DARM_RF_to_DC(),
                fa.FrequencyResponse3(
                    F_Hz,
                    [
                        (model.ETMX.p1.o, +fsig),
                        (model.ETMX.p1.o, -fsig),
                    ],
                    [
                        (model.ETMX.p1.o, +fsig),
                        (model.ETMX.p1.o, -fsig),
                    ],
                    name="gain",
                ),
            )
        )
        # gains = sol["gain"].out
        data = Munch(
            gains=sol.out,
            F_Hz=F_Hz,
            homs=model.homs,
            Parm_W=Parm_W,
        )
        save(fname, data)
    return data


# @pytest.mark.parametrize("ifo", ["lho", "llo"])
def T_all_optical_gains(optical_gains, tpath_join, ppath_join, fpath_join, makegrid):
    gains = optical_gains.gains
    F_Hz = optical_gains.F_Hz
    homs = optical_gains.homs
    orders = np.sum(homs, axis=1)
    maxtem = np.max(orders)
    print("maxtem", maxtem)

    def transfer(hom_idx):
        Hp = gains[..., 0, 0, hom_idx, hom_idx]
        Hl = gains[..., 1, 1, hom_idx, hom_idx]
        return Hp - Hl

    M_kg = 40
    cc = 8 * np.pi / (1064e-9 * scc.c * M_kg * (2 * np.pi)**2) * optical_gains.Parm_W

    def normalized_gain(hom_idx):
        Hsb = transfer(hom_idx)
        return -cc / F_Hz**2 * np.real(Hsb)

    plot_sign = True
    for order in range(0, maxtem + 1):
        print(order)
        oidx = np.nonzero(order == orders)[0]
        fig, ax = plt.subplots()
        axs = [ax]
        if plot_sign:
            fig_s, ax_s = plt.subplots()
            axs.append(ax_s)
        for idx in oidx:
            ax.semilogy(F_Hz, np.abs(normalized_gain(idx)), label=homs[idx])
            if plot_sign:
                ax_s.plot(F_Hz, np.sign(normalized_gain(idx)), label=homs[idx])
        for _ax in axs:
            makegrid(_ax, F_Hz)
            _ax.set_xlabel("Frequency [Hz]")
            # ax.set_ylabel("Normalized PI gain")
            # ax.set_title(f"{order} order modes")
            _ax.legend(ncol=2)
        ax.set_title(f"{order} order modes normalized PI gains")
        fig.savefig(tpath_join(f"gains{order}.pdf"))
        if plot_sign:
            ax_s.set_title(f"{order} order modes PI gain sign")
            fig_s.savefig(tpath_join(f"signs{order}.pdf"))
