# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import finesse
from finesse.utilities.maps import circular_aperture
from pathlib import Path

finesse.init_plotting()
base = Path("./figures/arm_cavity_losses")
base.mkdir(exist_ok=True, parents=True)


def DLCT(x1s, x2s=None, M_abcd=None, lam=1064e-9):
    A, B, C, D = M_abcd.ravel()

    if x2s is None:
        x2s = x1s

    if np.isclose(B, 0):
        L = np.diag(np.exp(-1j * np.pi * C / lam * x1s**2))
    else:
        dx1 = x1s[1] - x1s[0]
        x1g, x2g = np.meshgrid(x1s, x2s)

        arg = (A * x1g**2 - 2 * np.outer(x1s, x2s) + D * x2g**2) / (B * lam)
        L = dx1 * np.sqrt(1j / (B * lam)) * np.exp(-1j * np.pi * arg)
    return L


def CM_kernel(xs, C, lam=1064e-9, diag=False):
    """
    Chirp multiplication kernel
    """
    C = C / lam
    d = np.exp(-1j * np.pi * C * xs**2)
    if not diag:
        d = np.diag(d)
    return d


def q2w0(q, lam=1064e-9):
    """
    Get waist size from q parameter.
    """
    zr = np.imag(q)
    w0 = np.sqrt(zr * lam / np.pi)
    return w0


def q2w(q, lam=1064e-9):
    """
    Get beam size from q parameter.
    """
    w0 = q2w0(q, lam=lam)
    zr = np.imag(q)
    w = w0 * np.abs(q) / zr
    return w


def q_propag(q1, M, n1=1, n2=1):
    """
    Propagate a q parameter through an ABCD matrix
    """
    A, B, C, D = M.ravel()
    q2 = n2 * (A * q1 / n1 + B) / (C * q1 / n1 + D)
    return q2


def abcd_space(d, n=1):
    """
    ABCD matrix for free space of d meters
    """
    M = np.array([[1, d / n], [0, 1]])
    return M


def abcd_lens(p):
    """
    ABCD matrix for a lens with focal power of p diopters
    """
    M = np.array([[1, 0], [-p, 1]])
    return M


def abcd_mirror(R):
    """
    ABCD matrix for a mirror with radius of curvature R
    """
    return abcd_lens(2 / R)


def q_eig(M):
    """
    Computes both positive and negative solutions to the quadratic eigenmode equation
    -c*q**2 + (a-d)*q + b = 0
    """
    A, B, C, D = M.flatten()
    root_term = np.lib.scimath.sqrt(4 * B * C + (A - D) ** 2)
    q1 = ((A - D) + root_term) / (2 * C)
    q2 = ((A - D) - root_term) / (2 * C)
    if np.imag(q1) > 0:
        q = q1
    else:
        q = q2
    return q


# %%
T_ITM = 0.015
T_ETM = 0
R_ITM = 1 - T_ITM
R_ETM = 1 - T_ETM

t_ITM = np.sqrt(T_ITM)
t_ETM = np.sqrt(T_ETM)
r_ITM = np.sqrt(R_ITM)
r_ETM = np.sqrt(R_ETM)

RoC_ITM = 1934
RoC_ETM = 2245
len_ARM = 3994.45
TM_aperture_diam = 0.34
TM_aperture_radius = TM_aperture_diam / 2

md = abcd_space(len_ARM)
m1 = abcd_mirror(RoC_ITM)
m2 = abcd_mirror(RoC_ETM)

r_rt = r_ITM * r_ETM
m_rt = m1 @ md @ m2 @ md
q_cav = q_eig(m_rt)

# %%
results = {}
for N in [51, 101]:
    M = N + 1
    q_inc = q_cav  # input beam is perfectly mode matched
    q_to_ETM = q_propag(q_inc, md)
    w_ITM = q2w(q_inc)
    w_ETM = q2w(q_to_ETM)

    xs_ITM = np.linspace(-1, 1, N) * 4 * w_ITM
    ys_ITM = np.linspace(-1, 1, M) * 4 * w_ITM
    dx_ITM = xs_ITM[1] - xs_ITM[0]
    dy_ITM = ys_ITM[1] - ys_ITM[0]

    xs_ETM = np.linspace(-1, 1, N) * 4 * w_ETM
    ys_ETM = np.linspace(-1, 1, M) * 4 * w_ETM
    dx_ETM = xs_ETM[1] - xs_ETM[0]
    dy_ETM = ys_ETM[1] - ys_ETM[0]

    results[N] = []
    rs = np.linspace(0.15, 0.2, 5)

    for TM_aperture_radius in rs:
        print(N, TM_aperture_radius, end="\r")
        aperture_map_itm = circular_aperture(
            xs_ITM,
            ys_ITM,
            TM_aperture_radius,
        )
        aperture_map_etm = circular_aperture(
            xs_ETM,
            ys_ETM,
            TM_aperture_radius,
        )

        D_x_ITM_to_ETM = DLCT(xs_ITM, xs_ETM, M_abcd=md)
        D_y_ITM_to_ETM = DLCT(ys_ITM, ys_ETM, M_abcd=md)

        D_x_ETM_to_ITM = DLCT(xs_ETM, xs_ITM, M_abcd=md)
        D_y_ETM_to_ITM = DLCT(ys_ETM, ys_ITM, M_abcd=md)

        C_x_ETM = CM_kernel(xs_ETM, -2 / RoC_ETM, diag=True)
        C_y_ETM = CM_kernel(ys_ETM, -2 / RoC_ETM, diag=True)
        Rc_map_ETM = np.outer(C_y_ETM, C_x_ETM)

        C_x_ITM = CM_kernel(xs_ITM, -2 / RoC_ITM, diag=True)
        C_y_ITM = CM_kernel(ys_ITM, -2 / RoC_ITM, diag=True)
        Rc_map_ITM = np.outer(C_y_ITM, C_x_ITM)

        def op_rt_eig(v):
            """Round trip operator. Ignore mirror reflectivities for now."""
            Xin = np.reshape(v, [M, N])
            X1 = D_y_ITM_to_ETM @ Xin @ D_x_ITM_to_ETM.T
            X2 = aperture_map_etm * Rc_map_ETM * X1 * r_ETM
            X3 = D_y_ETM_to_ITM @ X2 @ D_x_ETM_to_ITM.T
            X4 = aperture_map_itm * Rc_map_ITM * X3 * r_ITM
            return np.ravel(X4)

        linop_rt_eig = scipy.sparse.linalg.LinearOperator(
            matvec=op_rt_eig, shape=[N * M, N * M]
        )

        # Can takes a while to run for large M,N
        N_eig = 20  # how many cavity eigenmodes to compute

        eh, ev = scipy.sparse.linalg.eigs(linop_rt_eig, k=N_eig)
        # sort the eigenmodes from lowest to highest loss
        idx = np.argsort(np.abs(eh))[::-1]

        eh_sorted = eh[idx]
        ev_sorted = ev[:, idx]
        em_sorted = np.transpose(np.reshape(ev_sorted, [M, N, N_eig]), [2, 0, 1])
        results[N].append(1 - np.abs(eh_sorted[0]) ** 2 - T_ITM)

    results[N] = np.array(results[N])

    plt.semilogy(rs / 1e-2, results[N] / 1e-6, label=N)

plt.legend(title="LCT dimension NxN")
plt.xlabel("Aperture radius [cm]")
plt.ylabel("HG00 loss [ppm]")
plt.title("HG00 loss vs aperture radius")
plt.savefig(base / "HG00_loss_vs_aperture_radius_LCT_dim.pdf")
# %%
np.savez(
    base / "HG00_loss_vs_aperture_radius_LCT_dim.npz",
    **{str(k): v for k, v in results.items()},
    rs=rs
)
