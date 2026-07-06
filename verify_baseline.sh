#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE_clean
ntu=60 st=r ve=shift le=ViT-B/32 nc=10 nepc=1700 ls=100 gpu=0 th=50 t=2 ss=5 tm=lb_ad_md
tdir="sk_feats/shift_${ss}_r/"; edir="sk_feats/shift_val_${ss}_r/"
wdir_1="results/${ss}_r"; wdir_2="results/${ss}_r_val"
echo "=== FS-VAE OFFICIEL ss=5 tm=lb_ad_md seed=5 (config papier) ==="
python train.py --ntu $ntu --ss $ss --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "neg_t" \
  --st $st --ve $ve --le $le --tm $tm --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
  --phase train --mode train --dataset $tdir --wdir $wdir_1 | tee verify_ss5.log
echo ">>> ZSL = $(grep -oP 'increased to \K[0-9.]+' verify_ss5.log | tail -1) (attendu ~85)"
