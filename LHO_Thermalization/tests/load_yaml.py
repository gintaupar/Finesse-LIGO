# A sample script to load a new state from a yaml file

import pickle, sys, finesse
from finesse.ligo.factory import ALIGOFactory

sys.path.append('/Users/spjadhav/WORK/Remote_Commissioning/ligo-commissioning-modeling/finesse-ligo-state-optimisation/src')
sys.path.append('/Users/spjadhav/WORK/Remote_Commissioning/ligo-commissioning-modeling/LLO')
from funcs import run_thermal_model, plot_thermal_evolution


suffix = 'test'
newyaml = 'llo_cold_state.yaml'

factory = ALIGOFactory("/Users/spjadhav/WORK/Remote_Commissioning/ligo-commissioning-modeling/LLO/yaml/llo_O4.yaml")
factory.update_parameters("/Users/spjadhav/WORK/Remote_Commissioning/ligo-commissioning-modeling/LLO/yaml/llo_addRH.yaml")
factory.update_parameters(newyaml)

run_thermal_model(factory,
                  maxtems=8,
                  datapath=f'thermal_evolution_{suffix}.pkl',
                  return_data_level=0,
                  update_yaml=newyaml,
                  w0=1.015e-3,
                  z=6.0,
                  t_evol=6000,
                  is_astig=True,
                  )

fulldata = []
with open(f'thermal_evolution_{suffix}.pkl', "rb") as file:
    data = pickle.load(file)

fulldata.append({'t': data['t'], 'outs': data['outs'], 'varyParam': '--'})

plot_thermal_evolution(fulldata, {'param': f'new yaml'}, plotfileloc=f'GA_state_powerup_{suffix}.pdf', axislims=False)
