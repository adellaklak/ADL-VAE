#!/bin/bash
# usage : ./submit_pool.sh "<cmd1>" "<cmd2>" ...   -> distribue sur les GPU du pool
B=/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE_clean
POOL=(5 7 11 19 23 29)
i=0
for CMD in "$@"; do
  N=${POOL[$((i % ${#POOL[@]}))]}
  echo "-> esterel$N : $CMD"
  oarsub -p esterel$N -l gpu=1,walltime=8:00:00 "cd $B && $CMD"
  i=$((i+1))
done
