#!/bin/bash

cd "$7"/src

# Run GA
echo ./GA_sampling.py --generation_id "$1" --population_size "$2" --top_x_percent "$3" --frac_crossover "$4" --frac_mutation "$5" --frac_shrink "$6" --repopath "$7" --suffix "$8"
./GA_sampling.py --generation_id "$1" --population_size "$2" --top_x_percent "$3" --frac_crossover "$4" --frac_mutation "$5" --frac_shrink "$6" --repopath "$7" --suffix "$8"
