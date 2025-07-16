# %%
import finesse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import dill
import glob as glob

finesse.init_plotting(fmts=["png"], dpi=100)


files = glob.glob("data/*.pkl")

fig1, axs1 = plt.subplots(1, 1, figsize=(7, 4))

for file in files:
    (
        solutions,
        P_RHs,
        start_time,
        ARM_POWER,
        LIMITING_APERTURE,
        ABS_COEFF,
        ROUGH_RH_ACT,
        ROUGH_SH_ACT,
    ) = dill.load(open(file, "rb"))

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
        figsize=(7, 6),
    )
    t_hr = solutions[0].t / 3600

    colors = cm.viridis(np.linspace(0, 1, len(P_RHs)))

    for sol, P_RH, color in zip(solutions, P_RHs, colors):
        ax1.plot(
            t_hr,
            sol.outputs["P_prc"],
            marker="o",
            markersize=2,
            label=f"{P_RH:.1f} W",
            color=color,
        )

        ax2.plot(
            t_hr,
            1 / sol.outputs["f"] / 1e-6,
            marker="o",
            markersize=2,
            color=color,
        )
    ax1.set_ylim(0, 130)
    ax1.set_ylabel("PRC gain [W/W]")
    ax1.set_title(
        f"PRC gain vs aperture radius\nP_ARM={ARM_POWER/1e3:.1f} kW AP={LIMITING_APERTURE/1e-3:.2f} mm"
    )

    ax2.set_ylabel("1/f [uD]")
    ax2.set_xlabel("Time [Hours]")

    sm = plt.cm.ScalarMappable(
        cmap=cm.viridis, norm=plt.Normalize(vmin=P_RHs.min(), vmax=P_RHs.max())
    )
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=[ax1, ax2], orientation="vertical", pad=0.1, location="right"
    )
    cbar.ax.set_position(
        [
            cbar.ax.get_position().x0 + 0.25,
            cbar.ax.get_position().y0,
            cbar.ax.get_position().width,
            cbar.ax.get_position().height,
        ]
    )
    cbar.set_label("P_RH [W]")

    plt.tight_layout()
    output_path = Path(__file__).parent / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / Path(__file__).with_suffix(f".{start_time}.pdf").name)

    final_values = [sol.outputs["P_prc"][-1] for sol in solutions]

    plt.figure(figsize=(7, 4))
    plt.plot(P_RHs, final_values, marker="o", linestyle="-")
    plt.xlabel("P_RH [W]")
    plt.ylabel("Final PRC gain [W/W]")
    plt.title(
        f"Final PRC gain vs ring heater power\nP_ARM={ARM_POWER/1e3:.1f} kW AP={LIMITING_APERTURE/1e-3:.2f} mm"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_path / Path(__file__).with_suffix(f".steady_state.{start_time}.pdf").name
    )

    plt.figure(1)
    plt.plot(
        P_RHs, final_values, marker="o", linestyle="-", label=f"{ARM_POWER/1e3:.1f} kW"
    )
    plt.xlabel("P_RH [W]")
    plt.ylabel("Final PRC gain [W/W]")
    plt.title("Final PRC gain vs ring heater power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

plt.figure(1)
plt.savefig(
    output_path / Path(__file__).with_suffix(f".steady_state.total.{start_time}.pdf").name
)