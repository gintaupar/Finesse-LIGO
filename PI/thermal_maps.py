import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ifo_thermal_state.aligo_3D import (
    make_test_mass_model,
    AdvancedLIGOTestMass3D,
)
from ifo_thermal_state.mesh import sweep_cylinder_3D
from ifo_thermal_state.math import composite_newton_cotes_weights


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


def make_ACO2_intensity(P_ACO2, fname):
    I_ACO2_data = np.loadtxt(fname, delimiter=" ")
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
        return I_ACO2_interp((x[0], x[1])) * P_ACO2

    return I_ACO2, I_ACO2_real


def get_opd(x, y, ss):
    z = np.linspace(-0.1, 0.1, 11)
    xyz, dT, mask = ss.evaluate_temperature(x, y, z, meshgrid=True)
    dT[~mask] = 0
    # Use better quadrature rule for integratiopn
    weights = composite_newton_cotes_weights(z.size, 5)
    dz = z[1] - z[0]
    OPD = (
        8.6e-06
        * dz
        * np.sum(
            dT[:, :, :, 0] * weights[None, None, :], axis=2
        )  # weight Z direction and sum
    )
    return OPD
