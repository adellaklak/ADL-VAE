#!/bin/bash
source /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/miniconda3/etc/profile.d/conda.sh
conda activate fsvae
unset LD_LIBRARY_PATH
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE_clean
ntu=60 st=r ve=shift le=ViT-B/32 nc=10 nepc=1700 ls=100 gpu=0 th=50 t=2 tm=lb_ad_md

for ss in 5 12; do
  tdir="sk_feats/shift_${ss}_r/"; edir="sk_feats/shift_val_${ss}_r/"
  WD1="results/basefull_${ss}_r"; WD2="${WD1}_val"
  rm -rf "$WD1" "$WD2"   # wdir vierge => run complet
  echo "===================== BASELINE ss=$ss ====================="

  echo "--- Stage 1 (train, ZSL) ---"
  python train.py --ntu $ntu --ss $ss --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "neg_t" \
    --st $st --ve $ve --le $le --tm $tm --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
    --phase train --mode train --dataset $tdir --wdir $WD1 | tee basefull_${ss}_stage1.log
  ZSL=$(grep -oP 'increased to \K[0-9.]+' basefull_${ss}_stage1.log | tail -1)

  echo "--- Stage 2 (val) ---"
  python train.py --ntu $ntu --ss $ss --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "neg_t" \
    --st $st --ve $ve --le $le --tm $tm --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
    --phase val --mode train --dataset $edir --wdir $WD2 | tee basefull_${ss}_stage2.log

  echo "--- Stage 3 (gating train) ---"
  r3=$(python gating_train.py --ntu $ntu --ss $ss --st $st --ve $ve --le $le --tm $tm \
    --phase val --dataset $edir --wdir $WD2 --th $th --t $t)
  echo "$r3" | tee basefull_${ss}_stage3.log
  thresh=$(echo "$r3" | grep -oP 'best threshold\s*\K[\d.]+' | head -1)
  temp=$(echo "$r3" | grep -oP 'best temperature\s*\K[\d.]+' | head -1)

  echo "--- Stage 4 (gating eval, H réel) ---"
  r4=$(python gating_eval.py --ntu $ntu --ss $ss --st $st --phase train --dataset $tdir \
    --wdir $WD1 --ve $ve --le $le --tm $tm --thresh $thresh --temp $temp)
  echo "$r4" | tee basefull_${ss}_stage4.log
  SA=$(echo "$r4" | grep -oP 'seen_accuracy:\s*\K[\d.]+' | head -1)
  UA=$(echo "$r4" | grep -oP 'unseen_accuracy:\s*\K[\d.]+' | head -1)
  HM=$(echo "$r4" | grep -oP 'h_mean:\s*\K[\d.]+' | head -1)

  echo ">>> BASELINE ss=$ss : ZSL=$ZSL | seen=$SA unseen=$UA H_real=$HM"
done
echo "===== BASELINE COMPLETE FINIE ====="
