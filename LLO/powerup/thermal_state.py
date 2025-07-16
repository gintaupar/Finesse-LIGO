import numpy as np
# from finesse.utilities.maps import circular_aperture
from finesse.ligo.maps import (
    get_test_mass_surface_profile_interpolated,
    aligo_O4_ESD_inner_aperture,
    aligo_O4_TM_aperture
    )
from types import SimpleNamespace
from finesse.knm import Map
from collections import defaultdict


class ThermalState:
    def __init__(self, factory, model, radius=0.17, N=201, axisymmetric = False):
        self.factory = factory
        self.model = model
        self.radius = radius
        self.N = N
        self.values = SimpleNamespace()
        self.previous_OPDs = defaultdict(list)
        self.external_OPDs = defaultdict(list)

        self.x, self.y = (
            np.linspace(-radius, radius, N),
            np.linspace(-radius, radius, N),
        )
        self.r = np.linspace(0, radius, N)
        self.x_r = np.hstack([-np.flip(self.r[1:]), (self.r)])
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2)
        
        
        _, self.TM_aperture = aligo_O4_TM_aperture(r_lim = radius, N = N)
        _, self.TMlens_aperture = aligo_O4_ESD_inner_aperture(r_lim = radius, N = N)
        
        self.initial_parameters = {"ITMX.Rcx": float(self.model.ITMX.Rcx),
                                   "ITMY.Rcx": float(self.model.ITMY.Rcx),
                                   "ETMX.Rcx": float(self.model.ETMX.Rcx),
                                   "ETMY.Rcx": float(self.model.ETMY.Rcx),
                                   "ITMXlens.f": float(self.model.ITMXlens.f),
                                   "ITMYlens.f": float(self.model.ITMYlens.f),
                                   }

        # Loop over each test mass and get the static surface errror
        # Commnet this out since this is not neccessary, let
        for TM in [model.ITMX, model.ITMY, model.ETMX, model.ETMY]:
            if "X" in TM.name:
                P = factory.params.X
            else:
                P = factory.params.Y
            TM._unfreeze()
            TM.static = 1 * get_test_mass_surface_profile_interpolated(
                P[TM.name[:-1]].ID, make_axisymmetric = axisymmetric
            )(self.x, self.y)
            TM.aperture = self.TM_aperture
            TM._freeze()
            self.external_OPDs[TM.name] = np.zeros_like(self.x_r)
            
        for TMlens in [model.ITMXlens, model.ITMYlens]:
            TMlens._unfreeze()
            TMlens.aperture = self.TMlens_aperture
            self.external_OPDs[TMlens.name] = np.zeros_like(self.x_r)

        self.add_detectors()

    @property
    def t(self):
        raise NotImplementedError()

    @property
    def dt(self):
        return self.dt

    def reset(self):
        pass

    def add_detectors(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def get_opd(self, name):
        raise NotImplementedError()

    def get_deformation(self, name):
        raise NotImplementedError()

    def get_mask(self, name):
        raise NotImplementedError()

    def update_initial_settings(self):
         self.initial_parameters = {"ITMX.Rcx": float(self.model.ITMX.Rcx),
                                   "ITMY.Rcx": float(self.model.ITMY.Rcx),
                                   "ETMX.Rcx": float(self.model.ETMX.Rcx),
                                   "ETMY.Rcx": float(self.model.ETMY.Rcx),
                                   "ITMXlens.f": float(self.model.ITMXlens.f),
                                   "ITMYlens.f": float(self.model.ITMYlens.f),
                                   }
        
    def update_maps(self):
        model = self.model
        model.beam_trace()

        for TM in [
            model.ITMX,
            model.ETMX,
            model.ITMY,
            model.ETMY,
        ]:
            TM.surface_map = Map(
                self.x,
                self.y,
                amplitude=self.TM_aperture,
                opd=TM.static + self.get_deformation(TM.name),
            )
            # self.previous_OPDs[TM.name].append(TM.surface_map.opd.copy())
            # Remove a weighted quadratic part of the curvature
            # spot_size = TM.p1.i.qx.w
            # a, _ = TM.surface_map.get_radius_of_curvature_reflection(spot_size)
            # # And add it to the original
            # TM.Rc = 2 / (2 / float(self.initial_parameters[TM.Rcx.full_name]) + 2 / a)
            # # print("!!", spot_size/1e-3)
            # TM.surface_map.remove_piston(spot_size)
            # TM.surface_map.remove_curvatures(spot_size)
            # TM.surface_map.remove_tilts(spot_size)
            # print(f"{TM.name} Rc: {float(TM.Rcx)} dioptre: {2/a/1e-6} [uD]", spot_size)

        for ITMlens in [model.ITMXlens, model.ITMYlens]:
            ITMlens.OPD_map = Map(
                self.x,
                self.y,
                amplitude=self.TMlens_aperture,
                opd=self.get_opd(ITMlens.name.strip("lens")) ,
            )
            # self.previous_OPDs[ITMlens.name].append(ITMlens.OPD_map.opd.copy())
            # a = ITMlens.OPD_map.get_thin_lens_f(
            #     ITMlens.p1.i.qx.w, average=True
            # )
            # ITMlens.f.value = 1 / (1/float(self.initial_parameters[ITMlens.f.full_name])+ 1 / a)
            # ITMlens.OPD_map.remove_piston(
            #     ITMlens.p1.i.qx.w
            # )
            # ITMlens.OPD_map.remove_curvatures(
            #     ITMlens.p1.i.qx.w, mode="average"
            # )
            # ITMlens.OPD_map.remove_tilts(ITMlens.p1.i.qx.w)
