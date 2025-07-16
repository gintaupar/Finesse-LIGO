# %%
import numpy as np
import matplotlib.pyplot as plt
import finesse
from pathlib import Path

finesse.init_plotting()
base = Path("./figures/arm_cavity_losses")
base.mkdir(exist_ok=True, parents=True)


# %%
fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
LCT = np.load("figures/arm_cavity_losses/HG00_loss_vs_aperture_radius_LCT_dim.npz")
FINESSE = np.load("figures/arm_cavity_losses/HG00_loss_vs_aperture_radius_map_dim.npz")

rs = LCT["rs"]
losses = LCT["101"]
plt.sca(axs[0])
plt.scatter(rs, losses / 1e-6, label="LCT")
plt.sca(axs[1])
plt.scatter(rs, losses / 1e-6, label="LCT")

rs = FINESSE["rs"]
losses = FINESSE["100"]
plt.sca(axs[0])
plt.plot(rs, losses / 1e-6, label="FINESSE", c="k")
plt.sca(axs[1])
plt.semilogy(rs, losses / 1e-6, label="FINESSE", c="k")

plt.sca(axs[0])
plt.legend()
plt.ylabel("HG00 loss [ppm]")
plt.title("HG00 loss vs aperture radius for LCT and FINESSE")

plt.sca(axs[1]) 
plt.xlabel("Aperture radius [cm]")
plt.ylabel("HG00 loss [ppm]")
plt.savefig()