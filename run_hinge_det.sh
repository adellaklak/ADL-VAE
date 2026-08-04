#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd "$(cd "$(dirname "$0")" && pwd)"
SS=$1; PW=${2:-0}; M=${3:-2.0}; SEED=${4:-5}; TM=${5:-lb_ad_md}; TAG=${6:-a}
ntu=60 st=r ve=shift le=ViT-B/32 nc=10 nepc=1700 ls=100 gpu=0
WD="results/hingeDET_${TM}_pw${PW}_m${M}_s${SEED}_${SS}_r_${TAG}"; rm -rf "$WD"
python train_hinge_det.py --ntu $ntu --ss $SS --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "neg_t" --seed $SEED \
  --st $st --ve $ve --le $le --tm $TM --use_pw $PW --margin $M \
  --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
  --phase train --mode train --dataset sk_feats/shift_${SS}_r/ --wdir "$WD" | tee hingeDET_${TM}_pw${PW}_m${M}_s${SEED}_${SS}_${TAG}.log
echo ">>> [hingeDET tag=$TAG pw=$PW m=$M seed=$SEED ss=$SS] ZSL = $(grep -oP 'increased to \K[0-9.]+' hingeDET_${TM}_pw${PW}_m${M}_s${SEED}_${SS}_${TAG}.log | tail -1)"
