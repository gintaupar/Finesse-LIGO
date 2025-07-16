import numpy as np
import ipdb

import finesse.analysis.actions as fa
import finesse.components as fc
from finesse.ligo.factory import aligo
from finesse.ligo.suspension import QUADSuspension
from wield.control.plotting import plotTF
from munch import Munch


def to_2p_vec(tfs_sb):
    """
    Convert from sideband to two-photon description for a vector

    Input should be sideband transfer functions (one for the upper and one for the
    conjugate lower for each HOM) from a single optical port to a single optical port
    as calculated with FrequencyResponse2 with at least 3 dimensions.
    The shape should be (..., 1, 2, nhoms)
    where nhoms is the number of HOMs in the model

    Returns the corresponding two-photon transfer functions with shape
    (..., 2 * nhoms, 2 * nhoms)
    """
    from scipy.linalg import block_diag
    assert(tfs_sb.shape[-3:-1] == (1, 2))
    nmodes = tfs_sb.shape[-1]
    tfs_sb2 = np.zeros(tfs_sb.shape[:-3] + (2 * nmodes,), dtype=complex)
    for idx_fr in range(nmodes):
        fr = slice(2 * idx_fr, 2 * (idx_fr + 1))
        tfs_sb2[..., fr] = tfs_sb[..., 0, :, idx_fr]
    tfs_sb2 = tfs_sb2.reshape(tfs_sb2.shape[:-1] + (1,) + (2 * nmodes,))
    A2i = np.array([
        [1, 1j],
        [1, -1j],
    ]) / np.sqrt(2)
    Ai = block_diag(*(nmodes * [A2i]))
    tfs_2p = tfs_sb2 @ Ai
    return tfs_2p


def transpose(M):
    return np.swapaxes(M, len(M.shape) - 1, len(M.shape) - 2)


def adjoint(M):
    return transpose(M).conjugate()


def Vnorm_sq(M):
    sq = adjoint(M) @ M
    assert(sq.shape[-2:] == (1, 1))
    return sq[..., 0, 0].real


def SQZ_metrics(tfs_sb):
    """
    Calculate the McCuller metrics

    Input
    -----
    tfs_sb: sideband transfer function as calculated with FrequencyResponse2
    """
    tfs_2p = to_2p_vec(tfs_sb)
    m_q = tfs_2p[..., 0, 0]
    m_p = tfs_2p[..., 0, 1]
    m_p2 = np.abs(m_p)**2
    m_q2 = np.abs(m_q)**2
    etaGamma = m_p2 + m_q2
    theta = 0.5 * np.angle((m_p + 1j * m_q) / (m_p - 1j * m_q))
    num = (m_p2 - m_q2)**2 + 4 * np.real(m_q * m_p.conjugate())**2
    den = 4 * (m_p2 + m_q2)**2
    Xi = 1/2 - np.sqrt(num / den)
    LOdotAS = Vnorm_sq(adjoint(tfs_2p))
    return Munch(
        theta=theta,
        etaGamma=etaGamma,
        Xi=Xi,
        lossMM=LOdotAS - etaGamma,
    )


def T_simple_DARM(tpath_join):
    print("")
    factory = aligo.ALIGOFactory("lho_O4.yaml")
    model_nosus = factory.make()
    # factory.options.QUAD_suspension_model = QUADSuspension
    factory.options.QUAD_suspension_model = fc.FreeMass
    factory.options.QUAD_suspension_kwargs = dict(mass=40)
    model_quad = factory.make()

    F_Hz = np.geomspace(0.1, 1e3, 300)

    def actions(key):
        return [
            fa.StoreModelAttr("DARM.DC", "SRCL.DC", name=f"attrs_{key}"),
            fa.DCFields(name=f"DC_{key}"),
            fa.FrequencyResponse(F_Hz, "DARM.AC.i", ["AS.DC", "AS45.Q"], name=f"fresp_{key}"),
        ]

    analysis = [aligo.InitialLock()]
    analysis.extend(actions("RF"))
    analysis.append(aligo.DARM_RF_to_DC())
    analysis.extend(actions("DC"))

    def calc_fresp(model, **modeopts):
        with model.temporary_parameters():
            model.modes(**modeopts)
            sol = model.run(fa.Series(*analysis))
        # ret = Munch(tf=sol["fresp"].out.squeeze(), dc=sol["DC"])
        ret = sol
        return ret

    print("no sus")
    nosus = calc_fresp(model_nosus, modes="off")
    print("planewave")
    planewave = calc_fresp(model_quad, modes="off")
    print("hom")
    hom = calc_fresp(model_quad, modes="even", maxtem=2)
    # ipdb.set_trace()
    # print(np.allclose(nosus.dc.fields, planewave.dc.fields)

    def plot_plant(key, idx, title, fname):
        fig = plotTF(F_Hz, nosus[f"fresp_{key}"].out[..., idx, 0], label="No sus")
        plotTF(
            F_Hz, planewave[f"fresp_{key}"].out[..., idx, 0], *fig.axes, ls="--",
            label="Planewave",
        )
        plotTF(
            F_Hz, hom[f"fresp_{key}"].out[..., idx, 0], *fig.axes, ls="-.",
            label="HOMs",
        )
        fig.axes[0].legend()
        fig.axes[0].set_title(title)
        fig.savefig(tpath_join(f"{fname}.pdf"))

    plot_plant("RF", 1, "RF Readout AS45", "RF_AS45")
    plot_plant("RF", 0, "RF Readout AS DC", "RF_ASDC")
    plot_plant("DC", 1, "DC Readout AS45", "DC_AS45")
    plot_plant("DC", 0, "DC Readout AS DC", "DC_ASDC")


def T_sensing_SQZ_metrics(tpath_join, makegrid, generate_figs):
    print("")
    factory = aligo.ALIGOFactory("lho_O4.yaml")
    factory.options.QUAD_suspension_model = fc.FreeMass
    factory.options.QUAD_suspension_kwargs = dict(mass=40)
    model = factory.make()
    model.modes("even", maxtem=6)
    F_Hz = np.geomspace(1, 10e3, 400)
    fsig = model.fsig.f.ref
    test_points = [
        (factory.SRM_AR_fr.i, +fsig),
        (factory.SRM_AR_fr.i, -fsig),
    ]
    sol = model.run(
        fa.Series(
            aligo.InitialLock(),
            aligo.DARM_RF_to_DC(),
            fa.FrequencyResponse(F_Hz, ["DARM.AC.i"], ["AS.DC"], name="fresp"),
            fa.FrequencyResponse2(F_Hz, test_points, ["AS.DC"], name="fresp2"),
        )
    )
    metrics = SQZ_metrics(sol["fresp2"].out)

    figs = generate_figs(*metrics.keys())
    figs.theta.semilogx(F_Hz, metrics.theta * 180 / np.pi)
    figs.lossMM.loglog(F_Hz, metrics.lossMM)
    figs.etaGamma.loglog(F_Hz, metrics.etaGamma)
    figs.Xi.loglog(F_Hz, metrics.Xi)
    figs.theta.set_title("SQZ rotation")
    figs.lossMM.set_title("Mismatch loss")
    figs.etaGamma.set_title("Quantum noise gain")
    figs.Xi.set_title("Dephasing")
    figs.theta.set_ylabel("Angle [deg]")
    for k, ax in figs.items():
        ax.set_xlabel("Frequency [Hz]")
        makegrid(ax, F_Hz)
        ax.figure.savefig(tpath_join(f"{k}.pdf"))

    fig = plotTF(F_Hz, sol["fresp"].out.squeeze())
    fig.axes[0].set_ylabel("Magnitude [W/m]")
    fig.axes[0].set_title("Sensing Function")
    fig.savefig(tpath_join("sensing_function.pdf"))
