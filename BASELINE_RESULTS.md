# Baseline FS-VAE — Résultats de référence

Config officielle : `lb_ad_md`, loss `original`, gating standard, seed=5 (hardcodée).
Features : MSF-GZSSAR officielles. GPU : esterel7 (best/12, variance 84.07–84.44).
Code non modifié (clone officiel wenhanwu95/FS-VAE).

| Split | ZSL  | H oracle | H réel (gating std) |
|-------|------|----------|---------------------|
| ss=5  | 84.4 | 86.5     | 63.97 (seen 56.66 / unseen 73.45) |
| ss=12 | 48.6 | 62.7     | 47.10 (seen 57.64 / unseen 39.82) |

Notes :
- Le H réel est plombé par le gating standard (« gating is the wall »).
- Le H oracle (routage parfait) montre le potentiel réel du modèle.
- Variance GPU sur ZSL ss=5 : 84.07–84.44 (non-déterminisme cuDNN malgré seed fixe).

Reproduire : `./run_baseline_full.sh` sur esterel7.
