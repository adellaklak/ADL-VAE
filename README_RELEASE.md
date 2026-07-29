# Release: meilleure config confirmée (hinge_asymmetric + lbac_md_bdavg_cvg)

## Résultat
- ZSL: 84.14 (ss=5) / 61.41 (ss=12)
- H réel (gating): 70.67 (ss=5) / 54.14 (ss=12)

## Commande exacte
```bash
python train_hinge.py --ntu 60 --ss <5|12> --margin 1.0 --use_pw 0 \
  --tm lbac_md_bdavg_cvg --st r --ve shift --le ViT-B/32 \
  --num_cycles 10 --num_epoch_per_cycle 1700 --latent_size 100 \
  --wdir results/hinge_lbac_md_bdavg_cvg_pw0_m1.0_s5_<ss>_r \
  --dataset sk_feats/shift_<ss>_r/ --phase train --mode train
```

## Contenu de cette release
- `train_hinge.py` : loss hinge asymétrique (branche `feature/hinge-asym-pw`)
- `sem_info/` : descriptions texte sources, dont `cvg_60.csv` (description contrastive)
  et `bd_raw_60.txt` (descriptions brutes par partie du corps, source de `bdavg`)
- `text_feats/ViT-B/32/{lbac,bdavg,ac,md}_60.npy` : embeddings CLIP déjà calculés

## Limite connue (transparence)
Le script exact qui transforme `bd_raw_60.txt` en `bdavg_60.npy` (agrégation par
partie du corps) n'a pas été retrouvé lors de l'audit du 29/07/2026. Les `.npy`
sont donc inclus tels quels (déjà générés) pour garantir la reproductibilité du
run, mais leur pipeline de génération reste à reconstituer/documenter si besoin
de les régénérer pour d'autres splits (ex: NTU-120).
