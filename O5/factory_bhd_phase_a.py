import numpy as np
from munch import Munch
import finesse
import finesse.components as fc
from finesse.detectors import MathDetector, PowerDetector, AmplitudeDetector
from finesse.symbols import Constant
import finesse.analysis.actions as fa

from finesse.ligo.factory import ALIGOFactory

class BHDPhaseAFactory(ALIGOFactory):
    """This is the "BHD Phase A" configuration of LIGO in O5. (It is still dc readout.)

    Nomenclature follows slide 7 of LIGO-G2401791-v1, except that optics labeled OMA*
    are left as OM* in this factory. (The OMB* optics are not installed for Phase A.)

    This layout is sometimes called option "5f" in Glasgow documents.
    """
    def add_output_path(self, model, OUTPUT, SRM_AR_fr, OMC_input_port):
        OFI = model.add(fc.DirectionalBeamsplitter("OFI"))

        OFIlens = model.add(
            fc.AstigmaticLens(
                "OFIlens",
                fx=OUTPUT.OFIlens.fx,
                fy=OUTPUT.OFIlens.fy,
                )
            )

        OM0 = model.add(
            fc.Beamsplitter(
                "OM0",
                T=OUTPUT.OM0.T,
                L=OUTPUT.OM0.L,
                Rc=OUTPUT.OM0.Rc,
                alpha=OUTPUT.OM0.AOI,
            )
        )        
        OM1 = model.add(
            fc.Beamsplitter(
                "OM1",
                T=OUTPUT.OM1.T,
                L=OUTPUT.OM1.L,
                Rc=OUTPUT.OM1.Rc,
                alpha=OUTPUT.OM1.AOI,
            )
        )
        OM2 = model.add(
            fc.Beamsplitter(
                "OM2",
                T=OUTPUT.OM2.T,
                L=OUTPUT.OM2.L,
                Rc=OUTPUT.OM2.Rc,
                alpha=OUTPUT.OM2.AOI,
            )
        )
        OM3 = model.add(
            fc.Beamsplitter(
                "OM3",
                T=OUTPUT.OM3.T,
                L=OUTPUT.OM3.L,
                Rc=OUTPUT.OM3.Rc,
                alpha=OUTPUT.OM3.AOI,
            )
        )
        model.connect(SRM_AR_fr, OFI.p1, L=OUTPUT.length_SRM_OFI)
        model.connect(OFI.p3, OFIlens.p1, L=OUTPUT.length_OFI_OFIlens)
        model.connect(OFIlens.p2, OM0.p1, L=OUTPUT.length_OFIlens_OM0)
        model.connect(OM0.p2, OM1.p1, L=OUTPUT.length_OM0_OM1)
        model.connect(OM1.p2, OM2.p1, L=OUTPUT.length_OM1_OM2)
        model.connect(OM2.p2, OM3.p1, L=OUTPUT.length_OM2_OM3)
        model.connect(OM3.p2, OMC_input_port, L=OUTPUT.length_OM3_OMC)

        # The AS telescopes need to be redone for BHD Phase A, so are commented out here.
        #self.add_AS_telescopes(model)

        self.AS_port = OM3.p3
        self.OFI_sqz_port = OFI.p2
