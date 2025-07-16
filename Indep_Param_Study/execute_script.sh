#!/bin/bash

cd /home/shreejit.jadhav/WORK/RC/ligo-commissioning-modeling/analysis/O4/Indep_Param_Study

# Run script
echo ./ITM_thermalization_study.py --index "$1" --p_itmx_Rc "$2" --p_itmy_Rc "$3" --p_min_itmx_f "$4" --p_max_itmx_f "$5" --p_min_itmy_f "$6" --p_max_itmy_f "$7" --N "$8" --yamlpath "$9" --suffix "${10}" 
./ITM_thermalization_study.py --index "$1" --p_itmx_Rc "$2" --p_itmy_Rc "$3" --p_min_itmx_f "$4" --p_max_itmx_f "$5" --p_min_itmy_f "$6" --p_max_itmy_f "$7" --N "$8" --yamlpath "$9" --suffix "${10}"
