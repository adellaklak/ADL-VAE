# Résultats — base propre (seed 5, features officielles MSF-GZSSAR, code officiel FS-VAE)

Métrique : accuracy ZSL. GPU fixe (pool esterel 5/7/11/19/23/29).
Bruit mesuré : ss=5 amplitude 0.37 (n=8) ; ss=12 amplitude 0.52, σ=0.21 (n=5).
Tout écart inférieur à ces valeurs n'est pas un signal.

## Baseline

| | ss=5 | ss=12 |
|---|---|---|
| ZSL | 84.44 | 48.6 |
| H oracle | 86.5 | 62.7 |
| H réel (gating std) | 63.97 | 47.10 |

## Descriptions (`train.py` officiel, argument `--tm`)

| tm | ss=5 | ss=12 | cos inter-classes |
|---|---|---|---|
| lb | 79.35 | 37.19 | 0.7974 |
| ad | 75.52 | 37.68 | 0.7897 |
| md | 82.89 | 44.46 | 0.8789 |
| ad_md | 83.55 | 45.29 | — |
| lb_ad_md (baseline) | 84.44 | 48.6 | — |
| lbac_md | 84.88 | 46.24 | — |
| **lbac_md_bdavg** | **85.47** | **51.22** | — |

`bdavg` seule : cos 0.9433 (la moins discriminante), et apporte +5.0 sur ss=12.
**Seul levier qui améliore les deux splits.**

## Loss : hinge asymétrique (`train_hinge.py`)

| marge | ss=5 | ss=12 |
|---|---|---|
| 1.0 | 82.23 | **54.89** |
| 2.0 | 82.15 | 54.31 |
| 4.0 | 82.30 | 54.28 |

Plateau sur la marge (écart 0.15 / 0.61) → mécanisme réel, pas artefact de réglage.
Gain +6.3 sur ss=12, coût −2.2 sur ss=5.

Multi-seed (marge 2.0), gain apparié sur ss=12 : seed 1 +5.32, seed 2 **+0.06**, seed 5 +5.71.
Le gain n'est pas robuste à la seed. Le coût sur ss=5 l'est (−2.51 / −2.95 / −2.29).

## RÉSULTAT NÉGATIF — le terme de confusion a un gradient nul

`train_confusion.py`, base loss = hinge_asym, même GPU (esterel7), ss=12 :

| impl | ss=5 | ss=12 |
|---|---|---|
| none | 81.93 | 54.59 |
| ema_nograd (code d'origine) | 82.01 | **54.59** |
| sample_grad (version différentiable) | 79.06 | 51.83 |

`none` et `ema_nograd` sont identiques à la décimale. Cause : les centroïdes EMA sont
calculés sous `torch.no_grad()` avec `.detach()`, donc `clamp(margin − ‖μᵢ−μⱼ‖)` est une
constante pour autograd. Le terme n'a jamais influencé l'optimisation.
Le gain attribué à « confusion-aware » venait du hinge asymétrique caché dans sa base_loss.
La version différentiable (`sample_grad`) dégrade les deux splits.

## RÉSULTAT NÉGATIF — l'alignement structuré est du bruit

`train_structure.py`, distance-matching entre prototypes texte et centroïdes skeleton :

| align_w | ss=5 | ss=12 |
|---|---|---|
| 0.5 | 82.82 | 48.56 |
| 1.0 | 82.74 | 49.60 |
| 2.0 | 83.55 | 47.40 |
| 4.0 | — | 46.64 |

Non monotone sur ss=5, pic isolé à w=1.0 sur ss=12 qui s'effondre de part et d'autre.
Profil de bruit, à opposer au plateau du hinge.

## Classifieur : Mahalanobis diagonale (`eval_maha.py`, zéro paramètre)

Règle `d = Σⱼ((zⱼ − μₖⱼ)/σₖⱼ)²`, `σ = exp(0.5·logvar)`, appliquée à un run entraîné.

| run | classifieur entraîné | cosine | Mahalanobis |
|---|---|---|---|
| baseline ss=5 | 84.44 | 81.05 | 65.71 |
| baseline ss=12 | 48.6 | 44.07 | **50.92** |
| hinge ss=5 | 82.23 | 80.01 | 76.33 |
| hinge ss=12 | 54.89 | 46.39 | 52.51 |

Bat le classifieur entraîné sur ss=12 (+2.3), s'effondre sur ss=5 (−18.7).
Dégrade le hinge → les deux ne sont pas orthogonaux.

## Configuration

| | ss=5 | ss=12 |
|---|---|---|
| α=0.5, 1700 ep, φ=30 (script officiel) | 84.44 | 48.6 |
| α=0.1 (Table 12 du papier) | 82.82 | 52.35 |
| 1900 epochs (Table 12) | 81.49 | 47.71 |
| φ=25 | 84.44 | 48.23 |
| φ=35 (Table 12) | 84.37 | 47.86 |

φ n'a aucun effet mesurable. Les 1900 epochs dégradent. α=0.1 suit le pattern habituel.

## Le pattern transversal

Cinq leviers indépendants, même signature — tout ce qui aide ss=12 nuit à ss=5 :

| levier | ss=5 | ss=12 |
|---|---|---|
| hinge | −2.2 | +6.3 |
| α=0.1 | −1.6 | +3.8 |
| structure w=1 | −1.7 | +1.0 |
| Mahalanobis | −18.7 | +2.3 |

Exception unique : les descriptions (`lbac_md_bdavg`, +1.03 / +2.6).

## Discriminance vs alignement

Cosinus inter-classes moyen (bas = plus discriminant) :

| comp | cos | perf |
|---|---|---|
| ad | 0.7897 | 75.52 / 37.68 |
| lb | 0.7974 | 79.35 / 37.19 |
| cvg | 0.8157 | — |
| cvg2 | 0.8193 | — |
| vg | 0.8534 | — |
| kd | 0.8582 | — |
| diff | 0.8583 | — |
| lbac | 0.8662 | — |
| tt | 0.8767 | — |
| md | 0.8789 | 82.89 / 44.46 |
| bc | 0.9023 | — |
| bdavg | 0.9433 | +5.0 en combinaison |

Relation inverse : les descriptions les plus séparées dans l'espace CLIP performent
le plus mal. Dans ce VAE cross-modal, l'alignement avec la géométrie squelette prime
sur la discriminance textuelle.

## Reproduire

Chaque contribution est isolée dans une branche, avec un diff minimal vs `train.py` officiel
(aucun import externe) :

| branche | fichier | contenu |
|---|---|---|
| `main` | `run_desc.sh` | ablation descriptions |
| `feature/hinge-asym-pw` | `train_hinge.py` | hinge asym + `--margin --use_pw --seed --tm` |
| `feature/confusion-aware` | `train_confusion.py` | 3 impl du terme de confusion |
| `feature/structure-align` | `train_structure.py` | distance-matching |
| `feature/mahalanobis` | `eval_maha.py` | règle d'inférence, applicable à tout run |
| `feature/freq-phi` | `model.py` | `split_freq` pilotable |
| `feature/desc-discr` | `sem_info/*.csv` | descriptions discriminantes + `gen_text_feat.py` |
| `feature/baseline-seeds` | `train_seed.py` | `train.py` + `--seed` |
