#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE_clean
SS=$1; WD=$2; TAG=$3
python eval_maha.py --ss $SS --wdir "$WD" | tee maha_${TAG}_${SS}.log
