import finesse
import finesse.components as fc
import finesse.detectors as fd

from finesse_fenics import AxisymmetricThermalFEA


def prc_model():
    model = finesse.Model()
    model.modes("even", maxtem=8)
    LASER = model.add(fc.Laser("LASER", P=1))
    PRM = model.add(fc.Mirror("PRM", T=0.031, L=0, Rc=-10.948))
    PR2 = model.add(fc.Beamsplitter("PR2", T=0, L=0, Rc=-4.543, alpha=0))
    PR3 = model.add(fc.Beamsplitter("PR3", T=0, L=0, Rc=36.021, alpha=0))
    LENS = model.add(fc.Lens("LENS", f=300e3))
    ITM = model.add(fc.Mirror("ITM", T=0, L=0, Rc=1940))

    model.link(
        LASER.p1,
        PRM.p2,
        PRM.p1,
        16608.6e-3,
        PR2.p1,
        PR2.p2,
        16093e-3 - 7e-3,
        PR3.p1,
        PR3.p2,
        19537.4e-3 + 4829.6e-3 + 40.0e-3,
        LENS,
        ITM.p2,
        ITM.p1,
    )

    model.add(fc.Cavity("PRC", ITM.p2.o, via=PRM.p1.o))
    model.beam_trace()
    LASER.p1.o.q = LASER.p1.o.q
    model.add(fd.PowerDetector("P_in", PRM.p2.i))
    model.add(fd.PowerDetector("P_prc", ITM.p2.i))
    model.add(fd.PowerDetector("P_itm", ITM.p1.o))
    model.add(fd.FieldDetector("E_itm", ITM.p1.i, f=0))
    model.add(fd.BeamPropertyDetector("q_in", PRM.p2.i, "q"))
    model.add(fd.BeamPropertyDetector("q_prc", ITM.p2.i, "q"))
    model.add(fd.BeamPropertyDetector("q_itm", ITM.p1.o, "q"))
    return model


def time_model(coating_radius=0.16):
    model = prc_model()
    model.add_parameter("t", 0)
    model.ITM.p1.i.q = finesse.BeamParam(w=0.053, Rc=1934)
    HEATER = model.add(fc.Laser("HEATER", P=100e3))
    model.link(HEATER.p1, model.ITM.p1)
    THERMAL = model.add(
        AxisymmetricThermalFEA(
            "THERMAL",
            model.ITM.p1,
            model.LENS,
            N_z=10,
            N_r=10,
            coating_radius=coating_radius,
        )
    )
    model.add(fd.MathDetector("P_abs", THERMAL.HR_absorbed_power.ref, dtype=float))
    model.add(fd.MathDetector("P_heater", HEATER.P.ref, dtype=float))
    model.add(fd.MathDetector("P_RH", THERMAL.P_RH.ref, dtype=float))
    model.add(fd.MathDetector("Rc", THERMAL.Rc.ref, dtype=float))
    model.add(fd.MathDetector("f", THERMAL.f.ref, dtype=float))
    model.add(fd.MathDetector("tuning", model.ITM.phi.ref, dtype=float))
    model.add(fd.ModeMismatchDetector("INPUT_MM", model.PRM.p2.i, model.PRM.p1.o))
    model.add(fd.ModeMismatchDetector("PRC_ARM", model.ITM.p2.i, model.ITM.p1.o))
    return model
