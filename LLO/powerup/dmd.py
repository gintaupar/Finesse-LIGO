from thermal_state import ThermalState

import numpy as np
from finesse.cymath.homs import HGModes
from functools import partial
from finesse.knm import Map
import h5py
from types import SimpleNamespace


def zero_initial_condition(x):
    return np.full((x.shape[1],), 0)


# Get intensity on HR surface:
def I_ITMX_HR(model, values, x):
    HGs = HGModes(model.ITMX.p1.i.q, model.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_itmx1"][:, None]
    E = np.sum(a, axis=0)
    intensity = (E * E.conj()).real
    return intensity * float(model.alpha_ITMX)


def I_ETMX_HR(model, values, x):
    HGs = HGModes(model.ETMX.p1.i.q, model.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_etmx"][:, None]
    E = np.sum(a, axis=0)
    intensity = (E * E.conj()).real
    return intensity * float(model.alpha_ETMX)


def I_ETMY_HR(model, values, x):
    HGs = HGModes(model.ETMY.p1.i.q, model.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_etmy"][:, None]
    E = np.sum(a, axis=0)
    intensity = (E * E.conj()).real
    return intensity * float(model.alpha_ETMY)


def I_ITMY_HR(model, values, x):
    HGs = HGModes(model.ITMY.p1.i.q, model.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_itmy1"][:, None]
    E = np.sum(a, axis=0)
    intensity = (E * E.conj()).real
    return intensity * float(model.alpha_ITMY)


class DMDThermalState(ThermalState):
    def __init__(
        self,
        *args,
        rom_data_file="./thermal_data/dmdc_operators_aligo_v0.h5",
        **kwargs,
    ):
        # raise NotImplementedError("Not working yet...")
        super().__init__(
            *args,
            **kwargs,
        )
        self.operators = {}

        with h5py.File(rom_data_file, "r") as hf:
            # OPD group:
            opd_group = hf.get("opd")
            self.operators["A_opd"] = opd_group.get("A")[:]
            self.operators["B_opd"] = opd_group.get("B")[:]
            self.operators["U_opd"] = opd_group.get("Ux")[:]
            # Deform group
            deform_group = hf.get("deform")
            self.operators["A_deform"] = deform_group.get("A")[:]
            self.operators["B_deform"] = deform_group.get("B")[:]
            self.operators["U_deform"] = deform_group.get("Ux")[:]
            # Control group
            control_group = hf.get("control")
            self.operators["U_I"] = control_group.get("Ux")[:]

        self.r = np.linspace(0, 0.17, 100)
        X, Y = np.meshgrid(self.x, self.y)
        self.R = np.sqrt(X**2 + Y**2)

        self.ts_itmx = SimpleNamespace()
        self.ts_itmy = SimpleNamespace()
        self.ts_etmx = SimpleNamespace()
        self.ts_etmy = SimpleNamespace()

        for ts_optic in [
            self.ts_itmx,
            self.ts_itmy,
            self.ts_etmx,
            self.ts_etmy,
        ]:
            ts_optic.dt = 20
            ts_optic.t = 0
            ts_optic.opd = [np.zeros_like(self.operators["U_opd"][:, 0])]
            ts_optic.x_opd = [np.zeros((self.operators["A_opd"].shape[1]))]
            ts_optic.deform = [np.zeros_like(self.operators["U_deform"][:, 0])]
            ts_optic.x_deform = [np.zeros((self.operators["A_deform"].shape[1]))]
            ts_optic.I = []
            ts_optic.uI = []

    @property
    def t(self):
        return self.ts_itmx.t

    @property
    def dt(self):
        return float(self.ts_etmx.dt.value)

    @dt.setter
    def dt(self, value):
        if value < 0:
            raise ValueError("dt > 0")

        if self.ts_etmx.dt.value != value:
            self.ts_itmx.dt.value = value
            self.ts_etmx.dt.value = value
            self.ts_itmy.dt.value = value
            self.ts_etmy.dt.value = value

    def reset(self):
        self.ts_itmx.t = 0
        self.ts_itmy.t = 0
        self.ts_etmx.t = 0
        self.ts_etmy.t = 0

        self.ts_itmx.set_initial_condition(zero_initial_condition)
        self.ts_itmy.set_initial_condition(zero_initial_condition)
        self.ts_etmx.set_initial_condition(zero_initial_condition)
        self.ts_etmy.set_initial_condition(zero_initial_condition)

    def step(self):
        def compute_new_opd_state(optic):
            """
            construct k+1 state of either OPD of eformation with
            state k & control k
            """
            # if dof == 'OPD':
            _A = self.operator["A_opd"]
            _B = self.operator["B_opd"]
            if optic.dt == 20:
                xk = optic.x_opd[-1]
                uk = optic.uI[-1]
                x_k1 = _A.dot(xk) + _B.dot(uk)
            elif optic.dt > 20:
                if optic.dt % 20 != 0:
                    optic.dt = ((optic.dt // 20) + 1) * 20
                    print(
                        f"Warning, time step set to the closest multiple of 20: {optic.dt} s"
                    )
                _nt = optic.dt // 20
                _xks = [optic.x_opd[-1]]
                uk = optic.uI[-1]
                for i in range(_nt):
                    x_k1 = _A.dot(_xks[i]) + _B.dot(uk)
                    _xks.append(x_k1)
                x_k1 = _xks[-1]
            optic.x_opd.append(x_k1.real)

        def compute_new_deformation_state(optic):
            """
            construct k+1 state of either OPD of eformation with
            state k & control k
            """
            _A = self.operators["A_deform"]
            _B = self.operators["B_deform"]
            if optic.dt == 20:
                xk = optic.x_deform[-1]
                uk = optic.uI[-1]
                x_k1 = _A.dot(xk) + _B.dot(uk)
            elif optic.dt > 20:
                if optic.dt % 20 != 0:
                    optic.dt = ((optic.dt // 20) + 1) * 20
                    print(
                        f"Warning, time step set to the closest multiple of 20: {optic.dt} s"
                    )
                _nt = optic.dt // 20
                _xks = [optic.x_deform[-1]]
                uk = optic.uI[-1]
                for i in range(_nt):
                    x_k1 = _A.dot(_xks[i]) + _B.dot(uk)
                    _xks.append(x_k1)
                x_k1 = _xks[-1]
            optic.x_deform.append(x_k1.real)

        self.update_maps()

    def get_opd(self, optic):
        x_u = optic.x_opd[-1]
        u = self.operators["U_opd"]
        _opd = np.linalg.multi_dot([u, x_u])
        optic.opd.append(_opd)
        return np.interp(self.R, self.r, _opd)

    def get_deformation(self, optic):
        x_u = optic.x_deform[-1]
        u = self.operators["U_deform"]
        _deform = np.linalg.multi_dot([u, x_u])
        optic.deform.append(_deform)
        return np.interp(self.R, self.r, _deform)

    def update_maps(self):
        self.model.beam_trace()

        for ts, TM in zip(
            [
                self.ts_itmx,
                self.ts_etmx,
                self.ts_itmy,
                self.ts_etmy,
            ],
            [
                self.model.ITMX,
                self.model.ETMX,
                self.model.ITMY,
                self.model.ETMY,
            ],
        ):
            TM.surface_map = Map(
                self.x,
                self.y,
                amplitude=get_mask(self.x, self.y, ts),
                opd=TM.static + get_deformation(self.x, self.y, ts),
            )
            # Remove a weighted quadratic part of the curvature
            spot_size = TM.p1.i.qx.w
            a, _ = TM.surface_map.get_radius_of_curvature_reflection(spot_size)
            # And add it to the original
            TM.Rc = 2 / (2 / float(self.initial_parameters[TM.Rcx.full_name]) + 2 / a)
            # print("!!", spot_size/1e-3)
            TM.surface_map.remove_curvatures(spot_size)
            TM.surface_map.remove_tilts(spot_size)
            # print(f"{TM.name} Rc: {float(TM.Rcx)} dioptre: {2/a/1e-6} [uD]", spot_size)

        self.model.ITMXlens.OPD_map = Map(
            self.x,
            self.y,
            amplitude=get_mask(self.x, self.y, self.ts_itmx),
            opd=get_opd(self.x, self.y, self.ts_itmx),
        )
        self.model.ITMXlens.f.value = self.model.ITMXlens.OPD_map.get_thin_lens_f(
            self.model.ITMXlens.p1.i.qx.w, average=True
        )
        self.model.ITMXlens.OPD_map.remove_curvatures(
            self.model.ITMXlens.p1.i.qx.w, mode="average"
        )
        self.model.ITMXlens.OPD_map.remove_tilts(self.model.ITMXlens.p1.i.qx.w)

        self.model.ITMYlens.OPD_map = Map(
            self.x,
            self.y,
            amplitude=get_mask(self.x, self.y, self.ts_itmx),
            opd=get_opd(self.x, self.y, self.ts_itmy),
        )
        self.model.ITMYlens.f.value = self.model.ITMYlens.OPD_map.get_thin_lens_f(
            self.model.ITMYlens.p1.i.qx.w, average=True
        )
        self.model.ITMYlens.OPD_map.remove_curvatures(
            self.model.ITMYlens.p1.i.qx.w, mode="average"
        )
        self.model.ITMYlens.OPD_map.remove_tilts(self.model.ITMYlens.p1.i.qx.w)

        self.model.beam_trace()

    def add_detectors(self):
        self.model.parse(
            """
            fd E_itmx1 ITMX.p1.i f=0
            fd E_itmx2 ITMX.p1.o f=0
            fd E_itmx3 ITMX.p2.i f=0
            fd E_itmx4 ITMX.p2.o f=0
            fd E_etmx  ETMX.p1.i f=0
                    
            fd E_itmy1 ITMY.p1.i f=0
            fd E_itmy2 ITMY.p1.o f=0
            fd E_itmy3 ITMY.p2.i f=0
            fd E_itmy4 ITMY.p2.o f=0
            fd E_etmy  ETMY.p1.i f=0
                    
            mathd Parm (Px+Py)/2

            fd E_refl_c0  IFI.p4.o f=0
            fd E_refl_u9  IFI.p4.o f=+f1
            fd E_refl_l9  IFI.p4.o f=-f1
            fd E_refl_u45 IFI.p4.o f=+f2
            fd E_refl_l45 IFI.p4.o f=-f2
                                
            fd E_prc_c0  PRM.p1.o f=0
            fd E_prc_u9  PRM.p1.o f=+f1
            fd E_prc_l9  PRM.p1.o f=-f1
            fd E_prc_u45 PRM.p1.o f=+f2
            fd E_prc_l45 PRM.p1.o f=-f2
                    
            fd E_src_c0  SRM.p1.o f=0
            fd E_src_u9  SRM.p1.o f=+f1
            fd E_src_l9  SRM.p1.o f=-f1
            fd E_src_u45 SRM.p1.o f=+f2
            fd E_src_l45 SRM.p1.o f=-f2
                    
            fd E_x_c0  ETMX.p1.i f=0
            fd E_x_u9  ETMX.p1.i f=+f1
            fd E_x_l9  ETMX.p1.i f=-f1
            fd E_x_u45 ETMX.p1.i f=+f2
            fd E_x_l45 ETMX.p1.i f=-f2
                    
            fd E_y_c0  ETMY.p1.i f=0
            fd E_y_u9  ETMY.p1.i f=+f1
            fd E_y_l9  ETMY.p1.i f=-f1
            fd E_y_u45 ETMY.p1.i f=+f2
            fd E_y_l45 ETMY.p1.i f=-f2
                    
            fd E_inx_c0  ITMXlens.p1.i f=0
            fd E_inx_u9  ITMXlens.p1.i f=+f1
            fd E_inx_l9  ITMXlens.p1.i f=-f1
            fd E_inx_u45 ITMXlens.p1.i f=+f2
            fd E_inx_l45 ITMXlens.p1.i f=-f2
                
            fd E_iny_c0  ITMYlens.p1.i f=0
            fd E_iny_u9  ITMYlens.p1.i f=+f1
            fd E_iny_l9  ITMYlens.p1.i f=-f1
            fd E_iny_u45 ITMYlens.p1.i f=+f2
            fd E_iny_l45 ITMYlens.p1.i f=-f2

            fd E_c0_as OM1.p1.i f=0
        """
        )
