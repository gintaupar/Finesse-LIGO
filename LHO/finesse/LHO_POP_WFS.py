# %%
import finesse.ligo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from finesse.analysis.actions import (
    SensingMatrixDC,
    OptimiseRFReadoutPhaseDC,
)
import finesse.components as fc
from finesse.ligo.factory import aligo
import sys
import finesse.analysis.actions as fac
from finesse.ligo.actions import InitialLockLIGO, DARM_RF_to_DC

finesse.init_plotting(fmts=['png'])

# %%
dark = True

#mpl.rcParams.update({
#    'figure.figsize': (7, 5),
#    'font.family': 'sans-serif',
#    'font.sans-serif': 'cm',
#})

if dark:
    plt.style.use('dark_background')
    plt.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['xkcd:azure', 'xkcd:orange', 'xkcd:neon green', 'xkcd:red', 'xkcd:pale purple', 'xkcd:canary yellow'])
    cmap_srm = mpl.colors.LinearSegmentedColormap.from_list(
        'srm', ['xkcd:bluegrey', 'xkcd:burgundy'])

    cmap_sr2 = mpl.colors.LinearSegmentedColormap.from_list(
        'sr2', ['xkcd:brownish grey', 'xkcd:silver', 'xkcd:dark turquoise'])

    cmap_sr3 = mpl.colors.LinearSegmentedColormap.from_list(
        'sr3', ['xkcd:blood orange', 'xkcd:ice'])
    
    def makegrid(ax):
        ax.grid(True, which='major', color='xkcd:grey', linewidth=3)
        ax.grid(True, which='minor', color='xkcd:grey', linewidth=3)
    
else:
    cmap_srm = mpl.colors.LinearSegmentedColormap.from_list(
        'srm', ['xkcd:bluegrey', 'xkcd:burgundy'])

    cmap_sr2 = mpl.colors.LinearSegmentedColormap.from_list(
        'sr2', ['xkcd:brownish grey', 'xkcd:silver', 'xkcd:dark turquoise'])

    cmap_sr3 = mpl.colors.LinearSegmentedColormap.from_list(
        'sr3', ['xkcd:dull orange', 'xkcd:dark'])
    
    def makegrid(ax):
        ax.grid(True, which='major', color='k', alpha=0.05,)
        ax.grid(True, which='minor', color='k', alpha=0.025,)

# %%
factory = aligo.ALIGOFactory(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_O4.yaml")

factory.options.ASC.add = True

model = factory.make()
model.modes(maxtem=4)
sol = model.run(fac.Series(InitialLockLIGO(), DARM_RF_to_DC()))

model.L0.P = 50
out = model.run()

# %%
def add_POP_WFS(model):
    # add beamsplitter to pop port
    model.add(fc.Beamsplitter('POP_BS', R=0.5, T=0.5))
    # connect BS to POP port
    model.connect(model.PR2.p3, model.POP_BS.p1)
    # create nothing spaces for POP WFS
    model.add(fc.Nothing('POP_A'))
    model.add(fc.Nothing('POP_B'))
    # connect nothing spaces to BS, set gouy phase by hand
    model.add(fc.Space('pop1', portA=model.POP_BS.p3, portB=model.POP_A.p1, user_gouy_x=0, user_gouy_y=0))
    model.add(fc.Space('pop2', portA=model.POP_BS.p2, portB=model.POP_B.p1, user_gouy_x=90, user_gouy_y=90))
    # add POP A readouts where gouy=0, x=yaw, y=pit, 36 and 45 MHz demods
    model.add(fc.ReadoutRF('POP_A_WFS36x', optical_node=model.POP_A.p2.o, f=model.f2-model.f1, pdtype='xsplit', output_detectors=True))
    model.add(fc.ReadoutRF('POP_A_WFS36y', optical_node=model.POP_A.p2.o, f=model.f2-model.f1, pdtype='ysplit', output_detectors=True))
    model.add(fc.ReadoutRF('POP_A_WFS45x', optical_node=model.POP_A.p2.o, f=model.f2, pdtype='xsplit', output_detectors=True))
    model.add(fc.ReadoutRF('POP_A_WFS45y', optical_node=model.POP_A.p2.o, f=model.f2, pdtype='ysplit', output_detectors=True))
    model.add(fc.ReadoutRF('POP_A_WFS9x', optical_node=model.POP_A.p2.o, f=model.f1, pdtype='xsplit', output_detectors=True))
    model.add(fc.ReadoutRF('POP_A_WFS9y', optical_node=model.POP_A.p2.o, f=model.f1, pdtype='ysplit', output_detectors=True))
    # add POP B readouts where gouy=90, x=yaw, y=pit, 36 and 45 MHz demods
    model.add(fc.ReadoutRF('POP_B_WFS36x', optical_node=model.POP_B.p2.o, f=model.f2-model.f1, pdtype='xsplit', output_detectors=True))
    model.add(fc.ReadoutRF('POP_B_WFS36y', optical_node=model.POP_B.p2.o, f=model.f2-model.f1, pdtype='ysplit', output_detectors=True))
    model.add(fc.ReadoutRF('POP_B_WFS45x', optical_node=model.POP_B.p2.o, f=model.f2, pdtype='xsplit', output_detectors=True))
    model.add(fc.ReadoutRF('POP_B_WFS45y', optical_node=model.POP_B.p2.o, f=model.f2, pdtype='ysplit', output_detectors=True))
    model.add(fc.ReadoutRF('POP_B_WFS9x', optical_node=model.POP_B.p2.o, f=model.f1, pdtype='xsplit', output_detectors=True))
    model.add(fc.ReadoutRF('POP_B_WFS9y', optical_node=model.POP_B.p2.o, f=model.f1, pdtype='ysplit', output_detectors=True))

    return model
# %%
def optimize_POP_WFS(model):
    model.run(OptimiseRFReadoutPhaseDC(
        'PRC2_P', 
        'POP_A_WFS36y_I',
        'PRC2_P',
        'POP_B_WFS36y_I',
        'PRC2_P', 
        'POP_A_WFS45y_I',
        'PRC2_P',
        'POP_B_WFS45y_I',
        'PRC2_P', 
        'POP_A_WFS9y_I',
        'PRC2_P',
        'POP_B_WFS9y_I',
        'PRC2_Y', 
        'POP_A_WFS36x_I',
        'PRC2_Y',
        'POP_B_WFS36x_I',
        'PRC2_Y', 
        'POP_A_WFS45x_I',
        'PRC2_Y',
        'POP_B_WFS45x_I',
        'PRC2_Y', 
        'POP_A_WFS9x_I',
        'PRC2_Y',
        'POP_B_WFS9x_I'))
    return model

def optimize_AS_WFS(model):
    model.run(
        OptimiseRFReadoutPhaseDC(
            "DHARD_P",
            "AS_A_WFS45y_Q",
            "DHARD_P",
            "AS_B_WFS45y_Q",
            "DHARD_Y",
            "AS_A_WFS45x_Q",
            "DHARD_Y",
            "AS_B_WFS45x_Q",
            "SRC1_P",
            "AS_A_WFS72y_Q",
            "SRC1_Y",
            "AS_A_WFS72x_Q",
            "MICH_P",
            "AS_A_WFS36y_Q",
            "MICH_Y",
            "AS_A_WFS36x_Q",
        )
    )

    return model

def optimize_REFL_WFS(model):
    model.run(
        OptimiseRFReadoutPhaseDC(
            "CHARD_P",
            "REFL_A_WFS45y_I",
            "CHARD_P",
            "REFL_B_WFS45y_I",
            "CHARD_Y",
            "REFL_A_WFS45x_I",
            "CHARD_Y",
            "REFL_B_WFS45x_I",
            "CHARD_P",
            "REFL_A_WFS9y_I",
            "CHARD_P",
            "REFL_B_WFS9y_I",
            "CHARD_Y",
            "REFL_A_WFS9x_I",
            "CHARD_Y",
            "REFL_B_WFS9x_I",
        )
    )
    return model
# %%
model = add_POP_WFS(model)
model = optimize_POP_WFS(model)
model.add(fc.DegreeOfFreedom('PRC3_P', model.PR3.dofs.pitch, +1))
model.add(fc.DegreeOfFreedom('PRC3_Y', model.PR3.dofs.yaw, +1))
# %%
dofs_P1 = (
    'PRC2_P', 'PRC3_P', 'MICH_P', 'CHARD_P', 'SRC1_P',
)
dofs_P2 = (
    'CSOFT_P', 'DHARD_P', 'SRC2_P', 'DSOFT_P'
)
dofs_Y1 = (
    'PRC2_Y', 'PRC3_Y', 'MICH_Y', 'CHARD_Y', 'SRC1_Y',
)
dofs_Y2 = (
    'CSOFT_Y', 'DHARD_Y', 'SRC2_Y', 'DSOFT_Y'
)

readouts_P = (
    'POP_A_WFS36y',
    'POP_B_WFS36y',
    'POP_A_WFS45y',
    'POP_B_WFS45y',
)
readouts_Y = (
    'POP_A_WFS36x',
    'POP_B_WFS36x',
    'POP_A_WFS45x',
    'POP_B_WFS45x',
)
# %%
sol1 = model.run(SensingMatrixDC(dofs_P1, readouts_P, d_dof=1e-10, name = 'pitch'))
sol4 = model.run(SensingMatrixDC(dofs_Y1, readouts_Y, d_dof=1e-10, name = 'yaw'))
# %%
fig_smat_dc, axs_smat_dc = sol1.plot(
    2,
    2,
    figsize=(9, 6),
    r_lims=4*[[0.1, 1e5]],
    )
leg_smat_dc = axs_smat_dc[-1].get_legend()
leg_smat_dc.set_bbox_to_anchor([0.5, 0.5], transform=fig_smat_dc.transFigure)
for ax in axs_smat_dc:
    ax.tick_params(axis='both', which='major', labelsize='small', pad=10)
    ax.set_title(ax.get_title().replace('_', ' '))
    ax.set_title(ax.get_title().replace('y', 'pit'))
    ax.title.set_size('large')
    makegrid(ax)

fig_smat_dc.tight_layout();
#fig_smat_dc.savefig('LHO/finesse/outputs/POP_WFS_sensing_pitch.png')

fig_smat_dc, axs_smat_dc = sol4.plot(
    2,
    2,
    figsize=(9, 6),
    #r_lims=4*[[0.1, 1e5]],
    )
leg_smat_dc = axs_smat_dc[-1].get_legend()
leg_smat_dc.set_bbox_to_anchor([0.5, 0.5], transform=fig_smat_dc.transFigure)
for ax in axs_smat_dc:
    ax.tick_params(axis='both', which='major', labelsize='small', pad=10)
    ax.set_title(ax.get_title().replace('_', ' '))
    ax.set_title(ax.get_title().replace('x', 'yaw'))
    ax.title.set_size('large')
    makegrid(ax)

fig_smat_dc.tight_layout();
#fig_smat_dc.savefig('LHO/finesse/outputs/POP_WFS_sensing_yaw.png')

# %%
# run 9 MHz simulation
readouts_P = (
    'POP_A_WFS9y',
    'POP_B_WFS9y',
)
readouts_Y = (
    'POP_A_WFS9x',
    'POP_B_WFS9x',
)

sol2 = model.run(SensingMatrixDC(dofs_P1, readouts_P, d_dof=1e-10, name = 'pitch'))
sol3 = model.run(SensingMatrixDC(dofs_Y1, readouts_Y, d_dof=1e-10, name = 'yaw'))

# %%
fig_smat_dc, axs_smat_dc = sol2.plot(
    1,
    2,
    figsize=(9, 6),
    r_lims=4*[[0.1, 1e5]],
    )
leg_smat_dc = axs_smat_dc[-1].get_legend()
leg_smat_dc.set_bbox_to_anchor([0.5, 0.5], transform=fig_smat_dc.transFigure)
for ax in axs_smat_dc:
    ax.tick_params(axis='both', which='major', labelsize='small', pad=10)
    ax.set_title(ax.get_title().replace('_', ' '))
    ax.set_title(ax.get_title().replace('y', 'pit'))
    ax.title.set_size('large')
    makegrid(ax)

fig_smat_dc.tight_layout();
#fig_smat_dc.savefig('LHO/finesse/outputs/POP_WFS_sensing_pitch9MHz.png')

fig_smat_dc, axs_smat_dc = sol3.plot(
    1,
    2,
    figsize=(9, 6),
    r_lims=4*[[0.1, 1e5]],
    )
leg_smat_dc = axs_smat_dc[-1].get_legend()
leg_smat_dc.set_bbox_to_anchor([0.5, 0.5], transform=fig_smat_dc.transFigure)
for ax in axs_smat_dc:
    ax.tick_params(axis='both', which='major', labelsize='small', pad=10)
    ax.set_title(ax.get_title().replace('_', ' '))
    ax.set_title(ax.get_title().replace('x', 'yaw'))
    ax.title.set_size('large')
    makegrid(ax)

fig_smat_dc.tight_layout();
#fig_smat_dc.savefig('LHO/finesse/outputs/POP_WFS_sensing_yaw9MHz.png')

# %%
