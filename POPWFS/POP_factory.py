import numpy as np

from munch import Munch
from finesse import Model
import finesse.components as fc
from finesse.detectors import PowerDetector
from finesse_ligo.factory import ALIGOFactory
from finesse.materials import FusedSilica

class POPWFSFactory(ALIGOFactory):
    def __init__(self, parameters, popwfs_params):
        # temporarily set POP WFS parameters separately until adding to a yaml file
        self.params = Munch()
        self.update_parameters(parameters)
        self.update_parameters(popwfs_params)
        self.reset()

    def set_default_options(self):
        super().set_default_options()

    def make(self):
        model = super().make()
        return model

    def add_PRC(self, model, params, BS_port, options):

        super().add_PRC(model, params, BS_port, options)

        PR2AR = model.add(
            fc.Mirror(
                "PR2AR",
                T=params.PR2AR.T,
                L=params.PR2AR.L,
                Rc=params.PR2AR.Rc,
            )
        )
        
        HAM3_mirror= model.add(
            fc.Beamsplitter(
                "HAM3_mirror",
                T= 50e-6, # AROM RH3, E1100510
                R= 1-50e-6,
                Rc=1.5,
            ))
    
        
        dichroic_BS = model.add(
            fc.Beamsplitter(
                "dichroic_BS",
                T=0, # splitter for 1064 nm POP and 532 nm ALS light
                L=0,
                Rc=np.inf,
            )
        )

        air_vac_BS = model.add(
            fc.Beamsplitter(
                "air_vac_BS",
                T= 0.054, # splitter for POPAIR and POPVAC paths
                R= 1-0.054, # 5.4% goes to vac path
                Rc=np.inf,
            )
        )

        pop_TT = model.add(
            fc.Beamsplitter(
                "pop_TT",
                T = 0, # tip tilt steering mirror for pop path
                L = 0, # no known parameters yet, except a flat mirror
                Rc = np.inf,
            )
        )

        pop_L1 = model.add(
            fc.Lens(
                "pop_L1", # tentative telescope solution
                f=334e-3, # proposed lens
            )
        )

        lsc_asc_BS = model.add(
            fc.Beamsplitter(
                "lsc_asc_BS", # tentative solution
                T= 0.5, # likely to be 50/50, but still under consideration
                R= 0.5,
                Rc=np.inf,
            )
        )

        pop_LSC = model.add(
            fc.Nothing("pop_LSC")
        )

        pop_ASC = model.add(
            fc.Nothing("pop_ASC")
        )
            
        model.connect(
                model.PR2.p4,
                PR2AR.p1,
                name="subPR2",
                L=params.PR2.thickness,
                nr=FusedSilica.nr,
            )

        model.connect(PR2AR.p2, HAM3_mirror.p1, name="PR2_HAM3", L=2.4483) # current distance
        model.connect(HAM3_mirror.p2, dichroic_BS.p1, name="HAM3_HAM1", L=19.3227) # possible new distance with HAM1 ISI, includes HAM3-HAM1, HAM1 periscope, peri to dichroic
        model.connect(dichroic_BS.p2, air_vac_BS.p1, name="dichroic_airvac", L=0.1016) # possible new distance with HAM1 ISI
        model.connect(air_vac_BS.p3, pop_TT.p1, name="airvac_TT", L=0.2286) # new distance, tentative layout
        model.connect(pop_TT.p2, pop_L1.p1, name="TT_L1", L=0.229) # confirmed LHO placement, 84307
        model.connect(pop_L1.p2, lsc_asc_BS.p1, name="L1_BS", L=0.178) # confirmed LHO placement, 84307
        model.connect(lsc_asc_BS.p2, pop_LSC.p1, name="BS_LSC", L=0.140) # confirmed LHO placement, 84313
        model.connect(lsc_asc_BS.p3, pop_ASC.p1, name="BS_ASC", L=0.200) # confirmed LHO placement, 84313

        self.POP_port = model.PR3.p2

    def add_ASC_readouts(self, model):
        super().add_ASC_readouts(model)

        f1 = model.f1.ref
        f2 = model.f2.ref
        output_detectors = self.options.ASC.add_output_detectors

        model.add(
            fc.ReadoutRF(
                "ASC_POP_A_36x",
                optical_node=model.pop_ASC.p1.i,
                f= f2 - f1,
                pdtype="xsplit",
                output_detectors=output_detectors
            )
        )

        model.add(
            fc.ReadoutRF(
                "ASC_POP_A_36y",
                optical_node=model.pop_ASC.p1.i,
                f= f2 - f1,
                pdtype="ysplit",
                output_detectors=output_detectors
            )
        )

        model.add(
            fc.ReadoutRF(
                "ASC_POP_A_45x",
                optical_node=model.pop_ASC.p1.i,
                f= f2,
                pdtype="xsplit",
                output_detectors=output_detectors
            )
        )

        model.add(
            fc.ReadoutRF(
                "ASC_POP_A_45y",
                optical_node=model.pop_ASC.p1.i,
                f= f2,
                pdtype="ysplit",
                output_detectors=output_detectors
            )
        )


POPWFS_params = Munch(PRC=Munch(
    PR2AR=Munch(
        Rc = np.inf,
        T = 1,
        L = 0,
    ),
    PR2= Munch(
        thickness = 75.1e-3
    ),
    PR3AR=Munch(
        Rc = np.inf,
        R = 0,
        L = 0,
        wedge = 0.57
    ),
    PR3=Munch(
        thickness = 101.62e-3
    ))
)