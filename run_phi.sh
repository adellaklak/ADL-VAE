#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd "$(cd "$(dirname "$0")" && pwd)"
SS=$1; PHI=$2
export SPLIT_FREQ=$PHI
ntu=60 st=r ve=shift le=ViT-B/32 nc=10 nepc=1700 ls=100 gpu=0 tm=lb_ad_md
WD="results/phi${PHI}_${SS}_r"; rm -rf "$WD"
python train.py --ntu $ntu --ss $SS --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "neg_t" \
  --st $st --ve $ve --le $le --tm $tm \
  --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
  --phase train --mode train --dataset sk_feats/shift_${SS}_r/ --wdir "$WD" | tee phi${PHI}_${SS}.log
echo ">>> [phi=$PHI ss=$SS] ZSL = $(grep -oP 'increased to \K[0-9.]+' phi${PHI}_${SS}.log | tail -1)"
