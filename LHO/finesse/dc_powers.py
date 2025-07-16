# %%
from finesse.ligo.lho import make_O4_lho
import numpy as np
from tabulate import tabulate
from tools import InitialLockLIGO

# %%
lho = make_O4_lho()
lho.modes("even", maxtem=4)
lho.L0.P = 60
# Need to make sure we're at a good operating point
# so we lock the LHO model
lho.run(InitialLockLIGO())

# %%
out = lho.run()
data = [
    ("P_x", out['Px']/1e3, 'kW'),
    ("P_y", out['Py']/1e3, 'kW'),
    ("PRG", out['PRG']),
    ("PRG9", out['PRG9']),
    ("PRG45", out['PRG45']),
    ("X arm gain", out['AGX']),
    ("Y arm gain", out['AGY']),
    ("P_REFL", out['Prefl'], 'W'),
    ("P_REFL", out['Prefl'], 'W'),
    ("P_PRC", out['Pprc'], 'W'),
    ("P_DCPD", out['Pas']/1e-3, 'mW')
]

print(tabulate(data, headers=["Name", "Value", "Unit"]))
# %%
