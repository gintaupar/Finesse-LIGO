# %% Fit to the last beam profile before going into the IFO

import numpy as np
import finesse
import finesse.ligo.lho
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys

finesse.init_plotting(fmts=['png'])

# %%
model = finesse.script.parse("""
# References
# [1] https://dcc.ligo.org/D1900436
# [2] https://galaxy.ligo.caltech.edu/optics/
# [3] https://alog.ligo-wa.caltech.edu/aLOG/index.php?callRep=60411
# [4] https://alog.ligo-wa.caltech.edu/aLOG/index.php?callRep=60356

squeezer SQZ 0
bs ZM4 T=0 R=1 Rc=13.33 alpha=0 # PSAMS
bs ZM5 T=0 R=1 Rc=3.4 alpha=0  # PSAMS
bs ZM6 Rc=inf R=1 T=0

link(
    SQZ,
    1, # Arbitrary distance
    ZM4,
    66.19*25.4m,
    ZM5,
    184.64*25.4m,
    ZM6,
)
""")

# %%
# Here we fit the squeezer mode, which is arbitrarily placed 1m before the ZM4 
# in this case, so that it matches the measured profile
# Data from https://alog.ligo-wa.caltech.edu/aLOG/index.php?callRep=60356

d_m = np.array([175, 95, 265, 568]) * 1e-3 - 175e-3
a1_4s_um = np.array([1770, 1730, 1835, 2055]) * 1e-6/2
a2_4s_um = np.array([1790, 1710, 1850, 2055]) * 1e-6/2


def f(w0, z0, d):
    q = finesse.BeamParam(z=z0, w0=w0)
    return q.beamsize(d-z0)

def f_opt(x, data):
    w = f(*x, d_m)
    err = np.sum(abs(data/1e-6 - w/1e-6)**2)
    return err

opt_x = minimize(f_opt, x0=(650e-6, -1), args=(a1_4s_um,))
opt_y = minimize(f_opt, x0=(650e-6, -1), args=(a2_4s_um,) )

qx = finesse.BeamParam(w0=opt_x.x[0], z=opt_x.x[1])
qy = finesse.BeamParam(w0=opt_y.x[0], z=opt_y.x[1])

print("Fitted data measured relative to ZM4 location")
print("Fitted qx:", qx)
print("Fitted qy:", qx)

if "pytest" not in sys.modules:
    plt.scatter(d_m, a1_4s_um/1e-6)
    plt.scatter(d_m, a2_4s_um/1e-6)
    plt.xlim(-1, 1)
    z = np.linspace(*plt.xlim())
    plt.plot(z, f(*opt_x.x, z)/1e-6, label=qx)
    plt.plot(z, f(*opt_y.x, z)/1e-6, label=qy)
    plt.legend(fontsize=8)
    plt.xlabel("Distance from ZM4 [m]")
    plt.ylabel("w [um]")
# %%
# The set PSAMs values at the time?
model.ZM4.Rc = 2/-113e-3
model.ZM5.Rc = 2/661e-3
model.ZM4.p1.i.qx = qx.reverse()
model.ZM4.p1.i.qy = qy.reverse()

bp = model.propagate_beam('SQZ.p1.o', 'ZM6.p1.i')
fig, ax = bp.plot_beamsizes(single_sided=True, show=False)
ax.scatter(bp.position(model.ZM4) + d_m, a1_4s_um/1e-3)
ax.scatter(bp.position(model.ZM4) + d_m, a2_4s_um/1e-3)
ax.set_xlim(0, 2)
ax.set_ylim(0.6, 1.2)
