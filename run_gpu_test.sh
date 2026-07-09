#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd "$(cd "$(dirname "$0")" && pwd)"
NODE=$1
ntu=60 st=r ve=shift le=ViT-B/32 nc=10 nepc=1700 ls=100 gpu=0 ss=5 tm=lb_ad_md
wdir="results/gputest_${NODE}_5_r"
rm -rf "$wdir"
START=$(date +%s)
python train.py --ntu $ntu --ss $ss --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "neg_t" \
  --st $st --ve $ve --le $le --tm $tm --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
  --phase train --mode train --dataset sk_feats/shift_${ss}_r/ --wdir "$wdir" | tee gputest_${NODE}.log
END=$(date +%s)
ELAPSED=$((END - START))
za=$(grep -oP 'increased to \K[0-9.]+' gputest_${NODE}.log | tail -1)
# écrire le temps dans un fichier dédié pour le récap
echo "$ELAPSED" > gputest_${NODE}.time
printf ">>> [%s] ZSL = %s | duree = %dh%02dm%02ds (%ds)\n" "$NODE" "$za" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
