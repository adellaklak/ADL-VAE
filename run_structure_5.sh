#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE_clean
ntu=60 st=r ve=shift le=ViT-B/32 nc=10 nepc=1700 ls=100 gpu=0 ss=5 tm=lb_ad_md
AW=${1:-1.0}
WD="results/structure_${ss}_w${AW}"
rm -rf "$WD"
echo "===== STRUCTURE ss=5 align_w=$AW ====="
python train_structure.py --ntu $ntu --ss $ss --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "struct" \
  --st $st --ve $ve --le $le --tm $tm --align_mode structure --align_w $AW \
  --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
  --phase train --mode train --dataset sk_feats/shift_${ss}_r/ --wdir "$WD" | tee structure_5_w${AW}.log
ZSL=$(grep -oP 'increased to \K[0-9.]+' structure_5_w${AW}.log | tail -1)
echo ">>> STRUCTURE ss=5 align_w=$AW : ZSL = $ZSL   (baseline ss=5 = 84.4)"
