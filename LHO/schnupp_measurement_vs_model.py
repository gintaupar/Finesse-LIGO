# %%
# These alogs detail some RF measurements of the Schnupp asymmetry
#   https://alog.ligo-wa.caltech.edu/aLOG/index.php?callRep=16084
#   https://alog.ligo-wa.caltech.edu/aLOG/index.php?callRep=16851
#
# They are essentially looking at the injecting an RF field into the IFO from
# the common side and measuring it at the AS port. Below I just misalign
# everything to get MICH and then vary the input laser frequency and see how
# much power makes it to AS_C.

from finesse_ligo.factory import ALIGOFactory
from finesse_ligo.actions import InitialLockLIGO, DARM_RF_to_DC
import finesse.analysis.actions as fac
import matplotlib.pyplot as plt
import finesse
import numpy as np

finesse.init_plotting()

# We first make a factory object that can generate an ALIGO model
# here we do so using the LHO O4 parameter file
factory = ALIGOFactory("lho_O4.yaml")

# Frequency (MHz)	 Refl PD power (dBm)	 AS PD power (dBm)
neg_data = np.array(
    [
        [-6.586e02, -2.319e01, -5.020e01],
        [-6.587e02, -2.319e01, -4.950e01],
        [-6.587e02, -2.330e01, -4.889e01],
        [-7.046e02, -2.400e01, -5.339e01],
        [-7.050e02, -2.400e01, -5.270e01],
        [-7.057e02, -2.389e01, -5.520e01],
        [-7.062e02, -2.400e01, -5.460e01],
        [-7.403e02, -2.419e01, -5.739e01],
        [-7.392e02, -2.419e01, -5.710e01],
        [-7.388e02, -2.410e01, -5.739e01],
        [-7.372e02, -2.400e01, -5.789e01],
        [-7.366e02, -2.400e01, -5.610e01],
        [-7.633e02, -2.389e01, -5.639e01],
        [-7.632e02, -2.400e01, -5.620e01],
        [-7.633e02, -2.400e01, -5.729e01],
        [-7.177e02, -2.339e01, -5.589e01],
        [-7.178e02, -2.339e01, -5.450e01],
        [-7.195e02, -2.339e01, -5.570e01],
        [-7.673e02, -2.389e01, -5.850e01],
        [-7.668e02, -2.389e01, -6.160e01],
        [-7.662e02, -2.389e01, -6.289e01],
        [-7.416e02, -2.430e01, -5.720e01],
        [-7.417e02, -2.430e01, -5.750e01],
        [-7.418e02, -2.430e01, -5.729e01],
        [-7.910e02, -2.489e01, -6.170e01],
        [-7.897e02, -2.500e01, -6.409e01],
        [-7.906e02, -2.489e01, -6.239e01],
        [-8.092e02, -2.480e01, -6.089e01],
        [-8.107e02, -2.519e01, -6.139e01],
        [-8.113e02, -2.469e01, -6.179e01],
        [-8.392e02, -2.550e01, -5.800e01],
        [-8.392e02, -2.560e01, -5.989e01],
        [-8.390e02, -2.560e01, -5.720e01],
        [-8.540e02, -2.680e01, -5.670e01],
        [-8.511e02, -2.539e01, -5.810e01],
        [-8.510e02, -2.550e01, -5.800e01],
        [-8.750e02, -2.539e01, -5.450e01],
        [-8.747e02, -2.530e01, -5.350e01],
        [-8.750e02, -2.530e01, -5.420e01],
        [-9.017e02, -2.580e01, -5.270e01],
        [-9.018e02, -2.580e01, -5.270e01],
        [-9.020e02, -2.580e01, -5.389e01],
        [-9.416e02, -2.639e01, -4.970e01],
        [-9.398e02, -2.650e01, -5.039e01],
        [-9.398e02, -2.639e01, -5.010e01],
        [-9.808e02, -2.760e01, -4.750e01],
        [-9.807e02, -2.710e01, -4.979e01],
        [-9.817e02, -2.710e01, -4.850e01],
        [-9.806e02, -2.710e01, -4.820e01],
        [-1.023e03, -2.789e01, -4.720e01],
        [-1.023e03, -2.789e01, -4.720e01],
        [-1.022e03, -2.789e01, -4.800e01],
        [-6.102e02, -2.239e01, -4.689e01],
        [-6.102e02, -2.239e01, -4.750e01],
        [-6.092e02, -2.250e01, -4.800e01],
        [-5.620e02, -2.169e01, -4.600e01],
        [-5.632e02, -2.180e01, -4.639e01],
        [-5.632e02, -2.169e01, -4.579e01],
    ]
)

pos_data = np.array(
    [
        [8.467e02, -2.560e01, -5.550e01],
        [8.477e02, -2.560e01, -5.789e01],
        [8.486e02, -2.550e01, -5.650e01],
        [9.612e02, -2.650e01, -5.829e01],
        [9.627e02, -2.660e01, -5.750e01],
        [9.633e02, -2.660e01, -5.829e01],
        [9.410e02, -2.639e01, -5.839e01],
        [9.361e02, -2.639e01, -6.120e01],
        [9.402e02, -2.639e01, -5.800e01],
        [9.413e02, -2.639e01, -6.100e01],
        [9.416e02, -2.630e01, -5.970e01],
        [9.220e02, -2.610e01, -7.240e01],
        [9.193e02, -2.610e01, -6.440e01],
        [9.198e02, -2.610e01, -6.579e01],
        [9.180e02, -2.600e01, -6.479e01],
        [9.002e02, -2.569e01, -6.289e01],
        [9.012e02, -2.569e01, -6.339e01],
        [9.005e02, -2.560e01, -6.329e01],
        [9.002e02, -2.689e01, -6.239e01],
        [8.737e02, -2.530e01, -6.200e01],
        [8.738e02, -2.539e01, -6.120e01],
        [8.740e02, -2.530e01, -6.120e01],
        [8.547e02, -2.530e01, -5.760e01],
        [8.542e02, -2.539e01, -5.789e01],
        [8.547e02, -2.530e01, -5.810e01],
        [8.548e02, -2.569e01, -5.800e01],
        [8.337e02, -2.550e01, -5.620e01],
        [8.346e02, -2.550e01, -5.650e01],
        [8.325e02, -2.539e01, -5.710e01],
        [8.317e02, -2.539e01, -5.589e01],
        [8.087e02, -2.480e01, -5.429e01],
        [8.122e02, -2.469e01, -5.450e01],
        [8.117e02, -2.469e01, -5.300e01],
        [7.911e02, -2.500e01, -5.139e01],
        [7.893e02, -2.489e01, -5.300e01],
        [7.902e02, -2.500e01, -5.339e01],
        [7.921e02, -2.500e01, -5.160e01],
        [7.703e02, -2.400e01, -4.989e01],
        [7.705e02, -2.400e01, -5.020e01],
        [7.705e02, -2.400e01, -5.029e01],
        [7.290e02, -2.369e01, -4.810e01],
        [7.285e02, -2.360e01, -4.820e01],
        [7.266e02, -2.360e01, -4.900e01],
        [6.622e02, -2.289e01, -4.529e01],
        [6.616e02, -2.289e01, -4.520e01],
        [6.737e02, -2.289e01, -4.529e01],
        [6.331e02, -2.250e01, -4.479e01],
        [6.336e02, -2.260e01, -4.400e01],
        [6.332e02, -2.260e01, -4.579e01],
        [6.326e02, -2.250e01, -4.460e01],
        [1.101e03, -2.960e01, -4.900e01],
        [1.101e03, -2.969e01, -4.839e01],
        [1.101e03, -2.930e01, -4.889e01],
        [1.054e03, -2.900e01, -5.070e01],
        [1.053e03, -2.860e01, -5.070e01],
        [1.052e03, -3.100e01, -5.129e01],
        [1.013e03, -2.760e01, -5.229e01],
        [1.013e03, -2.760e01, -5.279e01],
        [1.013e03, -2.760e01, -5.289e01],
        [9.893e02, -2.710e01, -5.450e01],
        [9.883e02, -2.700e01, -5.629e01],
        [9.890e02, -2.700e01, -5.579e01],
    ]
)

# %%
# Convert dBm to Watts
pos_data_watts = 10 ** ((pos_data[:, 2] - 30) / 10)
neg_data_watts = 10 ** ((neg_data[:, 2] - 30) / 10)
data_freq_offset = (pos_data[:, 0].min() - neg_data[:, 0].min()) / 2

plt.plot(
    pos_data[:, 0],
    pos_data_watts,
    "o",
    label="Positive schnupp",
)
plt.plot(
    neg_data[:, 0],
    neg_data_watts,
    "o",
    label="Negative schnupp",
)
plt.ylabel("Frequency [MHz]")
plt.xlabel("Power [W]")
plt.title("Data from alog 16084")

# %% The factory can now produce a model that can be used to run simulations
factory.reset()
lho = factory.make()
lho.modes("off")
lho.run(
    fac.Series(InitialLockLIGO(), DARM_RF_to_DC())
)  #  lock the model and run on DC readout

# %%
with lho.temporary_parameters():
    lho.ETMX.misaligned = True
    lho.ETMY.misaligned = True
    lho.PRM.misaligned = True
    lho.SRM.misaligned = True
    lho.MICH.DC += 45  # bright fringe

    for offset in [-1, 0.25, 1]:
        lho.ly1.L = 4.8478 - offset * 1e-2
        lho.lx1.L = 4.8296 + offset * 1e-2

        out = lho.run(fac.Xaxis("L0.f", "lin", 400e6, 1200e6, 100))
        plt.plot(
            out.x[0] / 1e6,
            out["Pas_c"] / out["Pas_c"].max(),
            label=f"$l_{{s}}$ = {lho.l_schnupp.eval()/1e-2:.2f} cm",
        )

# There is some offset in that was unexplained in alog 16084, the positive and
# negative frequency scans are not symmetric. The assumption was to just take
# the average of the minimums of the two scans.
plt.plot(
    pos_data[:, 0],
    pos_data_watts / pos_data_watts.max(),
    "x",
    label="Positive frequency",
    alpha=0.5,
)
plt.plot(
    -neg_data[:, 0],
    neg_data_watts / neg_data_watts.max(),
    "x",
    label="Negative frequency",
    alpha=0.5,
)
est_schnupp = (
    pos_data[np.argmin(pos_data_watts), 0]
    - neg_data[
        np.argmin(neg_data_watts),
        0,
    ]
) / 2

plt.axvline(
    est_schnupp,
    ls="--",
    color="k",
    label=f"Average minima (est. schnupp = {est_schnupp:.2f} MHz)",
),
plt.title("Input laser frequency offset vs schnupp asymmetry")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Power AS_C [arb. units]")
plt.legend(loc="upper right")
plt.savefig("figures/schnupp_vs_input_laser_frequency.pdf")

# %%
with lho.temporary_parameters():
    for offset in np.linspace(-2, 2, 10):
        lho.ly1.L = 4.8478 - offset * 1e-2
        lho.lx1.L = 4.8296 + offset * 1e-2

        out = lho.run()
        plt.figure(9)
        plt.scatter(lho.l_schnupp.eval() / 1e-2, out["PRG9"], c="r")
        plt.xlabel("Schnupp [cm]")
        plt.ylabel("Gain")
        plt.title("PRG9")

        plt.figure(45)
        plt.scatter(lho.l_schnupp.eval() / 1e-2, out["PRG45"], c="b")
        plt.xlabel("Schnupp [cm]")
        plt.ylabel("Gain")
        plt.title("PRG45")

plt.figure(45)
plt.axhline(30, label="Measurement mod.depth tests", ls="--", c="k")
plt.legend()
plt.savefig("figures/PRG45_vs_schnupp.png")
