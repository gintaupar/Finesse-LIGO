#!/bin/bash

cd "$4"/src

# Run simulations
echo ./GA_runsims.py --generation_id "$1" --idstart "$2" --idend "$3" --repopath "$4" --suffix "$5" --thermal_evol_time "$6" --cold_evol_time "$7"
./GA_runsims.py --generation_id "$1" --idstart "$2" --idend "$3" --repopath "$4" --suffix "$5" --thermal_evol_time "$6" --cold_evol_time "$7"
