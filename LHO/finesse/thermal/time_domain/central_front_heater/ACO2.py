# %%
import finesse
import finesse.ligo
import finesse.analysis.actions as fac
import numpy as np
import finesse.materials
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ifo_thermal_state.aligo_3D import (
    make_test_mass_model,
    AdvancedLIGOTestMass3D,
    AdvancedLIGOTestMass3DTime,
    AdvancedLIGOTestMass3DSteadyState
)
from ifo_thermal_state.mesh import sweep_cylinder_3D
from ifo_thermal_state.plotting import plot_mesh, plot_temperature, plot_deformation
import matplotlib.pyplot as plt
from finesse.knm import Map

from types import SimpleNamespace

def make_ACO2_intensity(values):
    I_ACO2_data = np.loadtxt("./CO2_Annular_Projection_I_1W.txt", delimiter=" ")
    _x = np.ascontiguousarray(I_ACO2_data[:512, 0])
    _y = np.ascontiguousarray(I_ACO2_data[::512, 1])
    _data = np.ascontiguousarray(I_ACO2_data[:,2].reshape((512, 512)))

    I_ACO2_interp = RegularGridInterpolator(
        (_x, _y), _data
    )

    r = np.linspace(0, 0.17, 100)
    phi = np.linspace(0, 2*np.pi, 50)
    R, PHI = np.meshgrid(r, phi)
    X = R * np.sin(PHI)
    Y = R * np.cos(PHI)

    I_ACO2_interp_sym = I_ACO2_interp((X.flat, Y.flat))
    I_ACO2_interp_sym = I_ACO2_interp_sym.reshape(X.shape)

    def I_ACO2(x):
        _r = np.sqrt(x[0]**2 + x[1]**2)
        return np.interp(_r, r, I_ACO2_interp_sym.mean(0))

    def I_ACO2_real(x):
        return I_ACO2_interp((x[0], x[1])) * values.P_ACO2
    
    return I_ACO2, I_ACO2_real

def make_CP_model():
    _CP = AdvancedLIGOTestMass3D()
    _CP.radius = 0.17
    _CP.thickness = 0.1
    return make_test_mass_model(
        mesh_function=sweep_cylinder_3D,
        mesh_function_kwargs={
            "num_elements": [8],
            "heights": [1],
            "add_flats": False,
            "HR_mesh_size": 0.015,
            "AR_mesh_size": 0.015,
            "mesh_algorithm": 6,
        },
        model=_CP
    )

def solve(CP, intensity_function):
    ss_aco2 = AdvancedLIGOTestMass3DSteadyState(CP)
    ss_aco2.temperature.I_HR.interpolate(intensity_function)
    ss_aco2.solve_temperature()
    ss_aco2.solve_deformation()
    return ss_aco2

# %%
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from tools import get_mask, get_deformation, get_opd

    finesse.init_plotting()
    CP = make_CP_model()
    # %%
    values = SimpleNamespace()
    values.P_ACO2 = 1
    I_ACO2, I_ACO2_real = make_ACO2_intensity(values)

    # %%
    ss_aco2 = solve(CP, I_ACO2)

    # %%
    # plot_temperature(ss_aco2.temperature.V, ss_aco2.temperature.solution)

    # %%
    # plot_deformation(ss_aco2.deformation.V, ss_aco2.deformation.solution)

    # %%
    x = y = np.linspace(-0.17, 0.17, 400)
    OPD = get_opd(x, y, ss_aco2)
    OPD[OPD != 0] -= OPD[200,200]
    f_opd = RegularGridInterpolator((x, y), OPD)

    # %%
    r = np.linspace(0, 0.17, 100)
    phi = np.linspace(0, 2*np.pi, 50)
    R, PHI = np.meshgrid(r, phi)
    X = R * np.sin(PHI)
    Y = R * np.cos(PHI)

    # %%
    rOPD = f_opd((X.flat, Y.flat))
    rOPD = rOPD.reshape(X.shape)
    # %%
    plt.plot(x, OPD[:, 200]/1e-6, label='x')
    plt.plot(x, OPD[200, :]/1e-6, label='y')
    plt.plot(r, rOPD.mean(0)/1e-6, label='mean')
    plt.xlabel("distance [m]")
    plt.ylabel("OPD [um]")
    plt.ylim(-0.03, 0.02)
    plt.legend()
    # %%

    np.savez_compressed("ACO2_ravg.npz", r=r, OPD=rOPD)
    # %%