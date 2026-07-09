#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE_clean
SS=$1; VARIANT=$2; IMPL=$3; CM=${4:-2.5}
ntu=60 st=r ve=shift le=ViT-B/32 nc=10 nepc=1700 ls=100 gpu=0 tm=lb_ad_md
TAG="${VARIANT}_${IMPL}"
WD="results/conf_${TAG}_${SS}_r"; rm -rf "$WD"
python train_confusion.py --ntu $ntu --ss $SS --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "neg_t" \
  --st $st --ve $ve --le $le --tm $tm --loss_variant $VARIANT --confusion_impl $IMPL --confusion_margin $CM \
  --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
  --phase train --mode train --dataset sk_feats/shift_${SS}_r/ --wdir "$WD" | tee conf_${TAG}_${SS}.log
echo ">>> [conf $TAG ss=$SS] ZSL = $(grep -oP 'increased to \K[0-9.]+' conf_${TAG}_${SS}.log | tail -1)"
