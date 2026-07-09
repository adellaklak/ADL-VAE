#!/bin/bash
# usage : ./submit_pool.sh <workdir> "<cmd1>" "<cmd2>" ...
WD=$(cd "$1" && pwd); shift
POOL=(5 7 11 19 23 29)
i=0
for CMD in "$@"; do
  N=${POOL[$((i % ${#POOL[@]}))]}
  echo "-> esterel$N : $WD :: $CMD"
  oarsub -p esterel$N -l gpu=1,walltime=8:00:00 "cd $WD && $CMD"
  i=$((i+1))
done
