# %%
# beam propagation cross check
import matplotlib.pyplot as plt
import finesse
import finesse.components as fc
import finesse.detectors as fd
import finesse_ligo
import numpy as np
from finesse.analysis.actions import For, Series, Execute, RunLocks, Noxaxis, SensingMatrixDC
from finesse_ligo.factory import ALIGOFactory

from finesse.tracing.tree import TraceTree

finesse.init_plotting()
# %%
factory = ALIGOFactory("lho_O4.yaml")

factory.options.INPUT.add_IMC_and_IM1 = True

lho = factory.make()
lho.fsig.f = 1
# make outgoing refl beam same AOI as ingoing beam from mode cleaner
lho.IM2_REFL.alpha = 7.0

# %%
# beam propagation from IMC to PRMAR
path = lho.path(lho.MC3.p3, lho.PRMAR.p2)
path.nodes
psx = lho.propagate_beam(lho.MC3.p3, lho.PRMAR.p2, direction='x')
psx.plot()
psx.qs
# %%
psy = lho.propagate_beam(lho.MC3.p3, lho.PRMAR.p2, direction='y')
psy.plot()
psy.qs
# %%
# beam propagation from IMC to REFL port at M5

path = lho.path(lho.MC3.p3, lho.M5.p1)
path.nodes
ps = lho.propagate_beam(lho.MC3.p3, lho.M5.p1)
ps.plot()
# %%
# propagate to PR3
psx = lho.propagate_beam(lho.MC3.p3, lho.PR3.p1, via_node = lho.ITMX.p2.i, direction='x')
psx.qs
psx.positions
# %%
# propagate to ITMX
psx = lho.propagate_beam(lho.MC3.p3, lho.PR3.p1.o, via_node = lho.ITMX.p2.i, direction='x')
psx.qs