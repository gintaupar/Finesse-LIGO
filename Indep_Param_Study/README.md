# Independent param study of ITM lenses and arm modes

```
repl() {
    sed "s/%%input_args_file%%/${2//\//\\/}/g" "$1"
}

export SUFFIX=20250129

mkdir -p data/data_${SUFFIX}
mkdir -p logs/logs_${SUFFIX}
mkdir -p jobs/jobs_${SUFFIX}

TEMPLATE=jobs/job_template.sub
ARG_FILE_PATH=$(pwd)/jobs/jobs_${SUFFIX}/args_${SUFFIX}.txt
CONDOR_JOBFILE=$(pwd)/jobs/jobs_${SUFFIX}/submit_jobs_${SUFFIX}.sub

repl ${TEMPLATE} ${ARG_FILE_PATH} > ${CONDOR_JOBFILE}

condor_submit ${CONDOR_JOBFILE}
```
