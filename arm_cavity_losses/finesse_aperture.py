# %%
import finesse
import numpy as np
from finesse.utilities.maps import circular_aperture
import matplotlib.pyplot as plt
from pathlib import Path

finesse.init_plotting()
base = Path("./figures/arm_cavity_losses")
base.mkdir(exist_ok=True, parents=True)

# %%
model = finesse.script.parse(
    """
    l l1 
    m ITM Rc=1934 T=0.015 L=0
    m ETM Rc=2245 T=0 L=0
    link(l1, ITM.p2, ITM.p1, 3994.45, ETM.p1)
    cav ARM ETM.p1.o
    """
)

model.modes(maxtem=4)
eigen_analysis = model.run("eigenmodes(ARM, 0)")
# Removing planewave loss takes out any loss due RTL on one roundtrip for a
# planewave - just leaves loss from HOM effects. Should be basically zero at
# this stage for every HOM
_, loss = eigen_analysis.loss(remove_planewave_loss=True)
print(loss)

# %%
print(model.ARM.info())

# %%
N = 1000
r = 0.17
x = np.linspace(-r, r, N)
y = np.linspace(-r, r, N + 1)
X, Y = np.meshgrid(x, y)

model.ITM.surface_map = model.ETM.surface_map = finesse.knm.Map(
    x,
    y,
    amplitude=circular_aperture(
        x,
        y,
        0.17,
    ),
)
eigen_analysis = model.run("eigenmodes(ARM, 0)")
_, loss = eigen_analysis.loss(remove_planewave_loss=True)

# Eigenvector index doesn't correspond to a HOM index
# The lowest loss is the fundamental mode though usually
idx = np.argsort(np.abs(loss))

plt.semilogy(idx, abs(loss / 1e-6))
plt.xlabel("Eigenvector idx")
plt.ylabel("HOM loss [ppm]")
plt.margins(x=0.1)
plt.title("Losses for different HOMs with aperture")


# %%
results = {}
for N in [100, 1000]:
    rs = np.linspace(0.15, 0.2, 10)
    results[N] = []

    for r in rs:
        x = np.linspace(-r, r, N)
        y = np.linspace(-r, r, N + 1)
        X, Y = np.meshgrid(x, y)

        aperture = finesse.knm.Map(
            x,
            y,
            amplitude=circular_aperture(
                x,
                y,
                r,
            ),
        )

        model.ITM.surface_map = aperture
        model.ETM.surface_map = aperture
        eigen_analysis = model.run("eigenmodes(ARM, 0)")
        _, loss = eigen_analysis.loss(remove_planewave_loss=True)
        results[N].append(np.min(np.abs(loss)))  # Get lowest loss as probably the HG00

    results[N] = np.array(results[N])
    plt.semilogy(rs / 1e-2, results[N] / 1e-6, label=N)
plt.legend(title="Map dimension NxN")
plt.xlabel("Aperture radius [cm]")
plt.ylabel("HG00 loss [ppm]")
plt.title("HG00 loss vs aperture radius")
plt.savefig(base / "HG00_loss_vs_aperture_radius_map_dim.pdf")


np.savez(
    base / "HG00_loss_vs_aperture_radius_map_dim.npz",
    **{str(k): v for k, v in results.items()},
    rs=rs
)


# %%
fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
for maxtem in [4, 6, 8, 10, 12, 14, 20]:
    model.modes(maxtem=maxtem)
    rs = np.linspace(0.15, 0.2, 10)
    losses = []

    for r in rs:
        N = 200
        x = np.linspace(-r, r, N)
        y = np.linspace(-r, r, N + 1)
        X, Y = np.meshgrid(x, y)

        aperture = finesse.knm.Map(
            x,
            y,
            amplitude=circular_aperture(
                x,
                y,
                r,
            ),
        )

        model.ITM.surface_map = aperture
        model.ETM.surface_map = aperture
        eigen_analysis = model.run("eigenmodes(ARM, 0)")
        _, loss = eigen_analysis.loss(remove_planewave_loss=True)
        losses.append(np.min(np.abs(loss)))  # Get lowest loss as probably the HG00

    losses = np.array(losses)
    plt.sca(axs[0])
    plt.plot(rs / 1e-2, losses / 1e-6, label=maxtem)
    plt.sca(axs[1])
    plt.semilogy(rs / 1e-2, losses / 1e-6, label=maxtem)

plt.sca(axs[0])
plt.legend(title="Max order", fontsize=8)
plt.ylabel("HG00 loss [ppm]")
plt.title("HG00 loss vs aperture radius")
plt.sca(axs[1])
plt.xlabel("Aperture radius [cm]")
plt.ylabel("HG00 loss [ppm]")
plt.savefig(base / "HG00_loss_vs_aperture_radius_maxtem.pdf")
