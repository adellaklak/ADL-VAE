#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd "$(cd "$(dirname "$0")" && pwd)"
SS=$1; TM=$2; WD1=$3   # WD1 = wdir du stage1 deja entraine
ntu=60 st=r ve=shift le=ViT-B/32 nc=10 nepc=1700 ls=100 gpu=0 th=50 t=2
tdir="sk_feats/shift_${SS}_r/"; edir="sk_feats/shift_val_${SS}_r/"
WD2="${WD1}_val"; rm -rf "$WD2"

echo "--- stage2 (val) ---"
python train_hinge.py --ntu $ntu --ss $SS --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "neg_t" \
  --st $st --ve $ve --le $le --tm $TM --use_pw 0 --margin 1.0 \
  --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
  --phase val --mode train --dataset $edir --wdir "$WD2" > /dev/null 2>&1

echo "--- stage3 (gating train) ---"
r3=$(python gating_train.py --ntu $ntu --ss $SS --st $st --ve $ve --le $le --tm $TM \
  --phase val --dataset $edir --wdir "$WD2" --th $th --t $t)
thresh=$(echo "$r3" | grep -oP 'best threshold\s*\K[\d.]+' | head -1)
temp=$(echo "$r3" | grep -oP 'best temperature\s*\K[\d.]+' | head -1)

echo "--- stage4 (gating eval, H reel) ---"
python gating_eval.py --ntu $ntu --ss $SS --st $st --phase train --dataset $tdir \
  --wdir "$WD1" --ve $ve --le $le --tm $TM --thresh $thresh --temp $temp \
  | tee gating_${TM}_${SS}.log
echo ">>> [H reel $TM ss=$SS] $(grep -oP 'h_mean:\s*\K[\d.]+' gating_${TM}_${SS}.log | head -1)"
