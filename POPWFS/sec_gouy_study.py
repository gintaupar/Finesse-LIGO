
'''Copied from HotStuff workshop repository
Written by E. Capote'''
# %%

import finesse
import finesse.ligo
from finesse_ligo import lho
from finesse.analysis.actions.sensing import SensingMatrixDC
import numpy as np
import matplotlib.pyplot as plt
from finesse.solutions import SimpleSolution
from finesse.analysis.actions import For, Series, Execute, RunLocks, Noxaxis
from finesse_ligo.factory import ALIGOFactory
from finesse_ligo.actions import InitialLockLIGO, DARM_RF_to_DC



finesse.init_plotting()

# %% Load in the parameter file
factory = ALIGOFactory("lho_O4.yaml")

factory.options.add_118MHz = True
factory.options.ASC.add = True
factory.options.ASC.close_AC_loops = False 
mass = 40  # mass is baked into the QUAD state-space model
model = factory.make()
model.fsig.f = 1

model.L0.P = 60

model.modes(maxtem=4)  # use at least up to 2nd order modes to include mismatch
model.run(InitialLockLIGO())
model.run(DARM_RF_to_DC())

# add pd at AS port to get DC power
model.parse(
"""
pd AS_A_pd node=AS_A.p1.i
pd AS_B_pd node=AS_B.p1.i
"""
)
# %%


dofs_P = (
    'SRC1_P', 'SRC2_P', 'MICH_P', 'DHARD_P'
)
dofs_Y = (
    'SRC1_Y', 'SRC2_Y', 'MICH_Y', 'DHARD_Y'
)

readouts_P = (
    'AS_A_WFS36y',
    'AS_A_WFS45y',
    'AS_A_WFS72y',
    'AS_B_WFS36y',
    'AS_B_WFS45y',
    'AS_B_WFS72y',
    'AS_Cy'
)
readouts_Y = (
    'AS_A_WFS36x',
    'AS_A_WFS45x',
    'AS_A_WFS72x',
    'AS_B_WFS36x',
    'AS_B_WFS45x',
    'AS_B_WFS72x',
    'AS_Cx'
)
# %%
for key in ['ls1', 'ls2', 'ls3']:
    space = model.get(key)
    space.user_gouy_x = 0
    space.user_gouy_y = space.user_gouy_x.ref
model.ls1.user_gouy_x = 20

# %%
def sensing_action(state, name):
    sol = SimpleSolution(name)
    sol0 = state.previous_solution
    sol.Psens = sol0['pitch'].matrix_data()[0]
    sol.Ysens = sol0['yaw'].matrix_data()[0]
    sol.DC_powerA = sol0['dc_power']['AS_A_pd']
    sol.DC_powerB = sol0['dc_power']['AS_B_pd']
    return sol

# %%
sec_gouy_deg = np.linspace(15, 165, 110)
model.modes(maxtem=1)
sol = model.run(
    For('ls1.user_gouy_x', sec_gouy_deg,
        Series(
            Series(
                RunLocks(exception_on_fail=False),
                SensingMatrixDC(dofs_P, readouts_P, d_dof=1e-10, name = 'pitch'),
                SensingMatrixDC(dofs_Y, readouts_Y, d_dof=1e-10, name = 'yaw'),
                Noxaxis(name='dc_power')
            ),
            Execute(sensing_action, name = 'sensing')
        )
    )

)
# %%
# makes radar plots, commented out to not generate 110 plots. 13 plots already generated and saved in ./ASC_plots/
#idx = 0
#for func in sol['series', 'series', 'pitch'].plot:
#    fig, axs = func(2, 2)
#    fig.suptitle('SEC gouy phase: ' + str(sec_gouy_deg[idx]) +' deg')
#    idx += 1


# %%
# pitch plots
dc_power = sol['series', 'sensing'].DC_powerA[0]
div_angle = np.array([
    model.SRM.p1.i.qx.divergence, model.SR2.p1.i.qx.divergence, model.BS.p1.i.qx.divergence, model.ETMX.p1.i.qx.divergence, 
    model.SRM.p1.i.qx.divergence, model.SR2.p1.i.qx.divergence, model.BS.p1.i.qx.divergence, model.ETMX.p1.i.qx.divergence
    ])

for i in range(len(readouts_P)-1):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    for j in range(len(dofs_P)):
        ax[0].semilogy(sec_gouy_deg, np.abs(sol['series', 'series', 'pitch'].out[:, j, i])*div_angle[j]/dc_power, label=dofs_P[j])
        ax[1].plot(sec_gouy_deg, np.angle(sol['series', 'series', 'pitch'].out[:, j, i], deg=True))
    ax[0].set_title(readouts_P[i])
    ax[0].set_ylabel('SNR')
    ax[0].legend()
    ax[1].set_ylabel('Phase')
    ax[1].set_xlabel('SEC Gouy Phase')
    #fig.savefig('/Users/elennac/hotstuff2023/sec_design/ASC_plots/'+readouts_P[i]+'_SECgouy.pdf')

fig, ax = plt.subplots()
for i in range(len(dofs_P)):
    ax.semilogy(sec_gouy_deg, np.abs(sol['series', 'series', 'pitch'].out[:, i, 6])*div_angle[i]/dc_power, label=dofs_P[i])
    ax.set_title('AS_Cy')
    ax.set_ylabel('SNR')
    ax.legend()
    ax.set_xlabel('SEC Gouy Phase')
#fig.savefig('/Users/elennac/hotstuff2023/sec_design/ASC_plots/AS_Cy_SECgouy.pdf')

# %%
# yaw plots
for i in range(len(readouts_Y)-1):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    for j in range(len(dofs_Y)):
        ax[0].semilogy(sec_gouy_deg, np.abs(sol['series', 'series', 'yaw'].out[:, j, i])*div_angle[j]/dc_power, label=dofs_Y[j])
        ax[1].plot(sec_gouy_deg, np.angle(sol['series', 'series', 'yaw'].out[:, j, i], deg=True))
    ax[0].set_title(readouts_Y[i])
    ax[0].set_ylabel('SNR')
    ax[0].legend()
    ax[1].set_ylabel('Phase')
    ax[1].set_xlabel('SEC Gouy Phase')
    #fig.savefig('/Users/elennac/hotstuff2023/sec_design/ASC_plots/'+readouts_Y[i]+'_SECgouy.pdf')

fig, ax = plt.subplots()
for i in range(len(dofs_Y)):
    ax.semilogy(sec_gouy_deg, np.abs(sol['series', 'series', 'pitch'].out[:, i, 6])*div_angle[i]/dc_power, label=dofs_Y[i])
    ax.set_title('AS_Cx')
    ax.set_ylabel('SNR')
    ax.legend()
    ax.set_xlabel('SEC Gouy Phase')
#fig.savefig('/Users/elennac/hotstuff2023/sec_design/ASC_plots/AS_Cx_SECgouy.pdf')
# %%
