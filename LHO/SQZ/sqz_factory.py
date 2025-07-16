import numpy as np

from munch import Munch
from finesse import Model
import finesse.components as fc
from finesse.detectors import PowerDetector
from finesse.ligo.factory import ALIGOFactory
from finesse.materials import FusedSilica

from substrate import Substrate


class LIGOwSQZFactory(ALIGOFactory):
    def __init__(self, parameters, sqz_params):
        # temporarily set SQZ parameters separately until adding to a yaml file
        self.params = Munch()
        self.update_parameters(parameters)
        self.update_parameters(sqz_params)
        self.reset()

    def set_default_options(self):
        super().set_default_options()
        self.options.add_ifo = False
        self.options.detailed_OFI = True
        self.options.SQZ.add = True
        self.options.SQZ.add_VIP = False
        self.options.SQZ.add_detectors = False
        self.options.SQZ.add_seed_laser = False

    def make(self):
        if self.options.add_ifo:
            model = super().make()
        else:
            model = Model()
            self.add_SRM(model, self.params.SRC)
            self.add_OMC(model, self.params.OMC)
            self.add_output_path(
                model, self.params.OUTPUT, self.SRM_AR_fr, self.OMC_input_port
            )
            self.add_squeeze_path(model, self.params.SQZ, self.OFI_sqz_port)
        self.post_make(model, self.params)
        return model

    def add_SRM(self, model, params):
        SRM = model.add(
            fc.Mirror("SRM", T=params.SRM.T, L=params.SRM.L, Rc=params.SRM.Rc)
        )
        SRMAR = model.add(
            fc.Mirror(
                "SRMAR",
                T=params.SRMAR.T,
                L=params.SRMAR.L,
                Rc=params.SRMAR.Rc,
                xbeta=SRM.xbeta.ref,
                ybeta=SRM.ybeta.ref,
                phi=SRM.phi.ref,
            )
        )
        model.connect(
            SRMAR.p2, SRM.p2, name="subSRM", L=params.SRM.thickness, nr=FusedSilica.nr
        )
        self._link_all_mechanical_dofs(SRM, SRMAR)
        self.SRM_AR_fr = SRMAR.p1
        self.SRM_HR_fr = SRM.p1

    def add_OMC(self, model, params):
        super().add_OMC(model, params)
        if self.options.add_detectors:
            model.add(PowerDetector("P_OMC", model.OMC_CM1.p2.o))

    def add_output_path(self, model, OUTPUT, SRM_AR_fr, OMC_input_port):
        OFI = model.add(fc.DirectionalBeamsplitter("OFI"))
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
        if self.options.detailed_OFI:
            L_TFP = 10e-3
            L_TGG = 25e-3
            L_QR = 5e-3
            L_FS = 5e-3
            L_KTP = 5e-3
            FS_KTP = 15e-3
            KTP_TGG = 150e-3
            TGG_QR = 50e-3
            QR_TFP = 220e-3
            L_OFI = L_TFP + L_TGG + L_QR + L_FS + L_KTP + FS_KTP + KTP_TGG + TGG_QR + QR_TFP
            FS = model.add(Substrate("OFI_FS", L=L_FS, nr=FusedSilica.nr))
            KTP = model.add(Substrate("OFI_KTP", L=L_KTP, nr=1.74))
            TGG = model.add(Substrate("OFI_TGG", L=L_TGG, nr=1.95))
            QR = model.add(Substrate("OFI_QR", L=L_QR, nr=FusedSilica.nr))
            TFP = model.add(Substrate("OFI_TFP", L=L_TFP, nr=FusedSilica.nr))
            model.connect(OFI.p3, FS.p1)
            model.connect(FS.p2, KTP.p1, L=FS_KTP)
            model.connect(KTP.p2, TGG.p1, L=KTP_TGG)
            model.connect(TGG.p2, QR.p1, L=TGG_QR)
            model.connect(QR.p2, TFP.p1, L=QR_TFP)
            model.connect(TFP.p2, OM1.p1, L=OUTPUT.length_OFI_OM1 - L_OFI)
        else:
            model.connect(OFI.p3, OM1.p1, L=OUTPUT.length_OFI_OM1)
        model.connect(OM1.p2, OM2.p1, L=OUTPUT.length_OM1_OM2)
        model.connect(OM2.p2, OM3.p1, L=OUTPUT.length_OM2_OM3)
        model.connect(OM3.p2, OMC_input_port, L=OUTPUT.length_OM3_OMC)

        self.add_AS_telescopes(model)

        self.AS_port = OM3.p3
        self.OFI_sqz_port = OFI.p2

    def post_make(self, model, params):
        pass


in2m = 0.0254
SQZ_params = Munch(SQZ=Munch(
    FC1=Munch(
        T=1e-3,
        Rc=np.inf,
        Rc_AR=1,
        thickness=78e-3,
    ),
    FC2=Munch(
        T=1e-6,
        Rc=534,
        Rc_AR=1,
        thickness=78e-3
    ),
    ZM1=Munch(
        T=0,
        L=0,
        Rc=np.inf,
        alpha=68,
    ),
    ZM2=Munch(
        T=0,
        L=0,
        Rc=0.85,
        # currently 3deg, should be reduced to 2deg if ZM1 is scooted, see D1900436
        alpha=3.2,
    ),
    ZM3=Munch(
        T=0,
        L=0,
        Rc=np.inf,
        alpha=27.5,
    ),
    ZM4=Munch(
        T=0,
        L=0,
        Rc=-2 / 150e-3,
        alpha=15.5,
    ),
    ZM5=Munch(
        T=0,
        L=0,
        Rc=3.4,
        alpha=5,
    ),
    ZM6=Munch(
        T=0,
        L=0,
        Rc=np.inf,
        alpha=118.5,
    ),
    length_FC=297.85,
    FC_loss=20e-6,
    length_ZM1_ZM2=59 * in2m,
    length_ZM2_ZM3=72 * in2m,
    length_ZM3_FC1=39 * in2m,
    length_ZM4_ZM5=66.19 * in2m,
    length_ZM5_ZM6=184.64 * in2m,
    length_ZM6_OFI=1150.756e-3 + 0.118,  # correction from Aidan
    # length_ZM6_OFI=1150.756e-3,
    length_VIP_ZM1=6.23 * in2m,
    length_VIP_ZM4=1086e-3,  # email from Don
    # length_VIP_ZM4=43.05 * in2m,
))
