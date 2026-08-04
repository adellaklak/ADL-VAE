#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd "$(cd "$(dirname "$0")" && pwd)"
SS=$1; WD=$2; TAG=$3
for epoch in 1699 3399 5099 6799 8499 10199 11899 13599 15299 16999; do
  echo "=== epoch $epoch ==="
  python eval_maha.py --ss $SS --tm lbac_md_bdavg_cvg --wdir "$WD" --epoch $epoch
done | tee maha_det_${TAG}_${SS}.log
