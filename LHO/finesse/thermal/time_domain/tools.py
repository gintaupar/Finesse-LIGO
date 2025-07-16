from ifo_thermal_state.math_utils import composite_newton_cotes_weights
import numpy as np



def get_mask(x, y, ss):
    _, _, mask = ss.evaluate_deformation(x, y, 0.1, meshgrid=True)
    return mask.astype(int)[:, :, 0]


def get_deformation(x, y, ss):
    xyz, S, mask = ss.evaluate_deformation(x, y, 0.1, meshgrid=True)
    S[~mask] = 0
    return S[:, :, 0, 2]


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