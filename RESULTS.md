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

## Descriptions (train.py officiel, --tm)

| tm | ss=5 | ss=12 |
|---|---|---|
| lb | 79.35 | 37.19 |
| ad | 75.52 | 37.68 |
| md | 82.89 | 44.46 |
| ad_md | 83.55 | 45.29 |
| lb_ad_md (baseline) | 84.44 | 48.6 |
| lbac_md | 84.88 | 46.24 |
| lbac_md_bdavg | 85.47 | 51.22 |

lbac_md_bdavg est le SEUL levier améliorant les deux splits (+1.03 / +2.6).
bdavg apporte +5.0 sur ss=12 (lbac_md 46.24 -> lbac_md_bdavg 51.22).

## Loss : hinge asymétrique (train_hinge.py)

| marge | ss=5 | ss=12 |
|---|---|---|
| 1.0 | 82.23 | 54.89 |
| 2.0 | 82.15 | 54.31 |
| 4.0 | 82.30 | 54.28 |

Plateau sur la marge -> mécanisme réel. Gain +6.3 ss=12, coût -2.2 ss=5.
Multi-seed (m=2.0), gain ss=12 apparié : seed 1 +5.32, seed 2 +0.06, seed 5 +5.71.
Gain NON robuste à la seed ; coût sur ss=5 robuste (-2.5 en moyenne).

## NÉGATIF — terme de confusion : gradient nul

train_confusion.py, base=hinge_asym, même GPU (esterel7) :

| impl | ss=5 | ss=12 |
|---|---|---|
| none | 81.93 | 54.59 |
| ema_nograd (code d'origine) | 82.01 | 54.59 |
| sample_grad (différentiable) | 79.06 | 51.83 |

none == ema_nograd (décimale). Centroïdes sous no_grad()+.detach() ->
clamp(margin - dist) constant -> gradient nul. Le gain attribué à "confusion"
venait du hinge caché dans base_loss. Version différentiable : dégrade.

## NÉGATIF — alignement structuré : bruit

train_structure.py :

| align_w | ss=5 | ss=12 |
|---|---|---|
| 0.5 | 82.82 | 48.56 |
| 1.0 | 82.74 | 49.60 |
| 2.0 | 83.55 | 47.40 |
| 4.0 | — | 46.64 |

Non monotone ss=5, pic isolé w=1.0 ss=12. Profil de bruit (vs plateau du hinge).

## Classifieur : Mahalanobis diag (eval_maha.py, zéro paramètre)

| run | classif. entraîné | cosine | Mahalanobis |
|---|---|---|---|
| baseline ss=5 | 84.44 | 81.05 | 65.71 |
| baseline ss=12 | 48.6 | 44.07 | 50.92 |
| hinge ss=5 | 82.23 | 80.01 | 76.33 |
| hinge ss=12 | 54.89 | 46.39 | 52.51 |

Bat le classifieur entraîné sur ss=12 (+2.3), s'effondre sur ss=5 (-18.7).
Dégrade le hinge -> pas orthogonaux.

## Configuration

| | ss=5 | ss=12 |
|---|---|---|
| script officiel (α=0.5, 1700ep, φ=30) | 84.44 | 48.6 |
| α=0.1 (Table 12 papier) | 82.82 | 52.35 |
| 1900 epochs (Table 12) | 81.49 | 47.71 |
| φ=25 | 84.44 | 48.23 |
| φ=35 (Table 12) | 84.37 | 47.86 |

φ sans effet mesurable. 1900 epochs dégrade. α=0.1 suit le pattern.

## Pattern transversal

Cinq leviers, même signature — aider ss=12 nuit à ss=5 :
hinge (-2.2/+6.3), α=0.1 (-1.6/+3.8), structure (-1.7/+1.0), Maha (-18.7/+2.3).
Exception : descriptions lbac_md_bdavg (+1.03/+2.6).

## Discriminance des descriptions

Cosinus inter-classes moyen (bas = plus discriminant), features skeleton brutes (256-d) :

| comp | cos inter | corr(dist_texte, dist_skel) | perf connue |
|---|---|---|---|
| ad | 0.7897 | 0.2067 | 75.52 / 37.68 |
| lb | 0.7974 | 0.1847 | 79.35 / 37.19 |
| cvg | 0.8157 | 0.1913 | — |
| cvg2 | 0.8193 | 0.2480 | — |
| vg | 0.8534 | 0.3306 | — |
| kd | 0.8582 | 0.2560 | — |
| diff | 0.8583 | 0.3497 | — |
| lbac | 0.8662 | 0.0707 | — |
| tt | 0.8767 | 0.3819 | — |
| md | 0.8789 | 0.3656 | 82.89 / 44.46 |
| bc | 0.9023 | 0.2274 | — |
| bdavg | 0.9433 | 0.1086 | +5.0 en combinaison |

CONSTAT : relation inverse entre discriminance et performance (les moins
discriminantes performent mieux : md 0.879, bdavg 0.943 gagnent ; ad 0.790,
lb 0.797 perdent).

NON EXPLIQUÉ : la corrélation des matrices de distances texte/skeleton (au niveau
entrée 256-d) ne prédit pas la performance. bdavg a la corrélation la plus basse
(0.109) mais le meilleur gain. tt a la plus haute (0.382) sans être bonne.
Le mécanisme derrière l'avantage des descriptions peu discriminantes reste à identifier.
Piste non testée : l'alignement dans le latent 100-d (post-encodage), non mesurable
sans entraînement.

## Reproduire

Une branche par contribution, diff minimal vs train.py officiel, zéro import externe :

| branche | fichier |
|---|---|
| main | run_desc.sh |
| feature/hinge-asym-pw | train_hinge.py (--margin --use_pw --seed --tm) |
| feature/confusion-aware | train_confusion.py (none/ema_nograd/sample_grad) |
| feature/structure-align | train_structure.py |
| feature/mahalanobis | eval_maha.py |
| feature/freq-phi | model.py (split_freq pilotable) |
| feature/desc-discr | sem_info/*.csv + gen_text_feat.py |
| feature/baseline-seeds | train_seed.py |

## COMBINAISON hinge + lbac_md_bdavg (résultat principal)

| config | ss=5 | ss=12 |
|---|---|---|
| baseline | 84.44 | 48.6 |
| hinge seul | 82.23 | 54.89 |
| lbac_md_bdavg seul | 85.47 | 51.22 |
| hinge + lbac_md_bdavg | 86.14 | 57.06 |

Les deux leviers s'additionnent (ss=12 : +8.5). Le coût du hinge sur ss=5 (-2.2)
DISPARAÎT sur les bonnes descriptions : 86.14 > baseline (84.44) ET > lbac_md_bdavg (85.47).
Le compromis ss=5/ss=12 observé sur chaque levier isolé n'existe plus en combinaison.
86.14 sur ss=5 dépasse le 86.9 du papier (features GPT-4 privées) à 0.76 près, en reproductible.
[seed 5 ; multi-seed à confirmer]

## Descriptions discriminantes seules (confirment la relation inverse)

| tm | cos inter | ss=5 | ss=12 |
|---|---|---|---|
| vg | 0.8534 | 77.73 | 30.64 |
| kd | 0.8582 | 71.83 | 34.46 |
| lbac_vg_kd | — | 77.65 | 36.85 |

Toutes sous md/lbac_md. lbac_vg_kd (77.65) << lbac_md (84.88) : le discriminant dégrade.

## Descriptions contrastives ciblées (cvg)

| config | ss=5 | ss=12 |
|---|---|---|
| lbac_md_bdavg | 85.47 | 51.22 |
| lbac_md_bdavg_cvg | 85.10 | 56.70 |
| lbac_md_bdavg_diff | 85.10 | 47.68 |
| cvg seul | 79.06 | 50.58 |

cvg (descriptions contrastives ciblant les paires confuses) apporte +5.5 sur ss=12
pour -0.37 sur ss=5, SANS modifier la loss. Proche de hinge+lbac_md_bdavg (57.06).
Contrairement aux discriminantes globales (vg/kd qui dégradent), cvg désambiguïse
explicitement les classes proches ("distinguishing it from eating..."). 
diff (autre contrastive) dégrade ss=12 (-3.5) : toutes les contrastives ne se valent pas.
Deux voies vers ~57 : hinge+lbac_md_bdavg (loss) OU lbac_md_bdavg_cvg (descriptions seules).

## Multi-seed de hinge+lbac_md_bdavg (robustesse)

| seed | baseline ss=12 | combo ss=12 | Δ | baseline ss=5 | combo ss=5 | Δ |
|---|---|---|---|---|---|---|
| 1 | 38.78 | 53.27 | +14.5 | 83.04 | 82.96 | -0.1 |
| 2 | 44.07 | 43.98 | -0.1 | 82.52 | 82.45 | -0.1 |
| 5 | 48.6 | 57.06 | +8.5 | 84.44 | 86.14 | +1.7 |

Gain ss=12 réel mais très variable (+14.5 / -0.1 / +8.5). Seed 2 = tirage où
ni hinge ni descriptions n'accrochent (déjà observé pour hinge seul : +0.06 sur seed 2).
Le résultat principal (86.14/57.06) est seed 5 = meilleur tirage, à annoncer comme tel.
Variance inter-seed ss=12 énorme (baseline : 38.78 à 48.6), à documenter en annexe.
