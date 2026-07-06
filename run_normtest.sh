#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE_clean
export NORM_PER_COMP=1
ntu=60 st=r ve=shift le=ViT-B/32 nc=10 nepc=1700 ls=100 gpu=0 ss=5 tm=lb_ad_md
python train.py --ntu $ntu --ss $ss --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "normpc" \
  --st $st --ve $ve --le $le --tm $tm --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
  --phase train --mode train --dataset sk_feats/shift_${ss}_r/ --wdir results/${ss}_r_normpc | tee normtest_ss5.log
echo ">>> NORM PAR COMPOSANTE ss=5 ZSL = $(grep -oP 'increased to \K[0-9.]+' normtest_ss5.log | tail -1) (baseline norm globale = 83.0)"
