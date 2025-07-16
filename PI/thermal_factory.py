from finesse.ligo.factory import aligo


class ALIGOThermalFactory(aligo.ALIGOFactory):
    def set_default_options(self):
        super().set_default_options()
        self.options.thermal.CO2 = None

    def add_test_mass_thermal(self, model):
        """
        Unlike with self absorption and RH, the CO2 power is set in the factory
        and cannot be changed in the model yet. FIX THIS.
        """
        super().add_test_mass_thermal(model)
        # I know, this is hacky
        try:
            P_CO2X = self.params.CO2.P_CO2X
            P_CO2Y = self.params.CO2.P_CO2Y
            CO2_data = self.params.CO2.CO2_data
        except AttributeError:
            return

        if self.options.thermal.CO2 == "annular" and (P_CO2X + P_CO2Y) > 0:
            print("importing slow stuff")
            from ifo_thermal_state.aligo_3D import AdvancedLIGOTestMass3DSteadyState
            import thermal_maps

            CP = thermal_maps.make_CP_model()

            def update_lens_map(lens, P_ACO2):
                I_ACO2, _ = thermal_maps.make_ACO2_intensity(P_ACO2, CO2_data)
                ss_aco2 = AdvancedLIGOTestMass3DSteadyState(CP)
                ss_aco2.temperature.I_HR.interpolate(I_ACO2)
                ss_aco2.solve_temperature()
                OPD_map = lens.OPD_map
                ACO2_SUB_1W = thermal_maps.get_opd(OPD_map.x, OPD_map.y, ss_aco2)
                opd = OPD_map.opd + P_ACO2 * ACO2_SUB_1W
                lens.OPD_map = Map(
                    OPD_map.x, OPD_map.y,
                    amplitude=OPD_map.amplitude,
                    opd=opd,
                )

            if P_CO2X > 0:
                update_lens_map(model.ITMXlens, P_CO2X)
            if P_CO2Y > 0:
                update_lens_map(model.ITMYlens, P_CO2Y)

    def post_make(self, model, params):
        super().post_make(model, params)
        model.SRCL.DC = 90
