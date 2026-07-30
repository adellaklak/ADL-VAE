#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/wt_t2m-augment
SS=$1; TM=$2
le=ViT-B/32 ntu=60 st=r ls=100 gpu=0
WD="results/augment_${TM}_${SS}_r"; rm -rf "$WD"
python train_augment.py --ntu $ntu --ss $SS --st $st --le $le --tm $TM \
  --latent_size $ls --gpu $gpu --dataset sk_feats/shift_${SS}_r/ \
  --frozen_ckpt frozen_ckpt/${SS}/se_*.pth.tar --wdir "$WD" \
  | tee augment_${TM}_${SS}.log
echo ">>> [augment tm=$TM ss=$SS] ZSL = $(grep -oP 'increased to \K[0-9.]+' augment_${TM}_${SS}.log | tail -1)"
git add augment_${TM}_${SS}.log
git commit -m "log: augment run tm=${TM} ss=${SS}" -q 2>/dev/null || true
git push origin feature/text-to-motion-augment -q 2>/dev/null || true
