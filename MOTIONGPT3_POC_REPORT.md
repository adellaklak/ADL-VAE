# MotionGPT3 → ShiftGCN : preuve de concept sur les classes unseen (ss=12)

**Date** : 2026-08-11
**Contexte** : GZSL-SAR, FS-VAE, NTU RGB+D 60, split ss=12. Piste explorée : générer
des squelettes synthétiques via MotionGPT3 (text-to-motion) pour les classes unseen,
comme alternative/complément au bruit gaussien conditionné-texte utilisé actuellement
pour entraîner le classifieur unseen.

## Objectif de cette étape

Avant d'investir dans le retargeting complet et l'intégration au VAE, valider une
question simple : les embeddings ShiftGCN extraits de squelettes MotionGPT3 générés
atterrissent-ils dans une zone cohérente de l'espace de features — proches de la
vraie classe, avec des confusions qui ont un sens — ou le signal est-il dégénéré ?

## Pipeline construit

1. **Génération** — MotionGPT3 (checkpoint `motiongpt3.ckpt`, HuggingFace
   `OpenMotionLab/motiongpt3`), sortie `(T, 22, 3)`, topologie HumanML3D/SMPL
   22-joints. 3 variantes de prompt par classe (debout/neutre, assis quand
   physiquement plausible, label court).
2. **Retargeting 22 → 25 joints (NTU)** — correspondance anatomique dérivée
   nous-mêmes (voir ci-dessous). **Pas** une recette tirée d'un papier externe :
   une citation trouvée en session prétendant fournir cette correspondance
   (arXiv 2604.21668, annexe C.1) s'est révélée fabriquée après lecture complète
   du papier réel — aucun contenu de ce type n'y figure. Leçon retenue : dériver
   soi-même plutôt que faire confiance à une citation non vérifiée.
3. **Recentrage** — chaque séquence (réelle et générée) recentrée sur son propre
   joint racine avant tout traitement.
4. **Mise en forme** — zero-pad à 300 frames (convention confirmée empiriquement
   de `NTU60_CS.npz`, pas 64 comme initialement supposé — bug trouvé et corrigé
   en cours de route, cf. section Bugs).
5. **Extraction** — ShiftGCN gelé (poids `sk_feats/shift_12_r/weights/best.pt`,
   entraîné uniquement sur les 48 classes seen, protocole ZSL inductif standard).
   Réimplémentation CPU pure de l'opération shift (`shift_pure.py`, sans
   dépendance CUDA) pour contourner le mur d'infrastructure déjà documenté
   ailleurs dans le projet.
6. **Comparaison** — similarité cosinus entre chaque embedding généré et les 12
   centroïdes réels unseen (moyenne des `ztest.npy` par classe, `shift_12_r`).

## Correspondance de joints (NTU 25 ← HumanML3D/SMPL 22)

```python
NTU_FROM_H3D = {
    0: 0,   1: 6,   2: 12,  3: 15,  4: 16,  5: 18,  6: 20,  7: 20,
    8: 17,  9: 19,  10: 21, 11: 21, 12: 1,  13: 4,  14: 7,  15: 10,
    16: 2,  17: 5,  18: 8,  19: 11, 20: 9,  21: 20, 22: 20, 23: 21, 24: 21,
}
```

19 correspondances directes. 6 articulations NTU sans équivalent (HandLeft/Right,
HandTip L/R, Thumb L/R) faute pointer sur le poignet correspondant — la
représentation HumanML3D 22-joints s'arrête au poignet, aucune info doigts.

## Bugs trouvés et corrigés en cours de route

- Sortie MotionGPT3 avec une dimension batch superflue `(1, T, 22, 3)` non gérée
  au premier essai → écrasait l'animation entière en une seule "frame".
- Confusion initiale entre classes unseen ss=12 (8 réellement 1-personne + 4
  classes 2-personnes hors-scope) et classes unseen ss=5-seulement (reading,
  writing, hat, jump up — seen pour ss=12, donc sans centroïde ss=12) : 4 des 12
  variantes testées n'avaient pas de cible valide, faussant le premier comptage.
- Écart d'échelle spatiale majeur non traité au premier essai : `z` généré proche
  de 0 (convention HumanML3D canonicalisée), `z` réel (fichier `.skeleton` brut)
  dans une tout autre plage — mais `NTU60_CS.npz` (le fichier réellement consommé
  par le pipeline) s'est révélé déjà recentré sur la racine, rendant la comparaison
  initiale non pertinente.
- Longueur de séquence : le modèle attend 300 frames zero-paddées (confirmé par
  un sanity-check sur un vrai échantillon, similarité 0.931 / rang 1 après
  correction), pas un rééchantillonnage à 64 frames comme tenté initialement —
  cause principale de la dégénérescence du tout premier résultat (18/22
  prédictions convergeant à tort sur une seule classe).

## Résultat (après correction des bugs ci-dessus)

| classe | variante | nom | sim→vraie | prédite | rang vraie | OK |
|---|---|---|---|---|---|---|
| 3 | standing | brush hair | 0.629 | 12 | 2 | non |
| 3 | seated | brush hair | 0.710 | 3 | 1 | **OUI** |
| 3 | short_label | brush hair | 0.637 | 40 | 2 | non |
| 5 | standing | pick up | 0.773 | 42 | 2 | non |
| 5 | short_label | pick up | 0.451 | 40 | 11 | non |
| 9 | standing | clapping | 0.453 | 40 | 8 | non |
| 9 | seated | clapping | 0.458 | 40 | 12 | non |
| 9 | short_label | clapping | 0.400 | 56 | 11 | non |
| 12 | standing | tear up paper | 0.540 | 15 | 6 | non |
| 12 | seated | tear up paper | 0.722 | 3 | 2 | non |
| 12 | short_label | tear up paper | 0.626 | 56 | 2 | non |
| 15 | standing | put on a shoe | 0.691 | 42 | 6 | non |
| 15 | seated | put on a shoe | 0.654 | 42 | 4 | non |
| 15 | short_label | put on a shoe | 0.630 | 5 | 8 | non |
| 40 | standing | sneeze/cough | 0.761 | 40 | 1 | **OUI** |
| 40 | seated | sneeze/cough | 0.725 | 58 | 2 | non |
| 40 | short_label | sneeze/cough | 0.693 | 15 | 2 | non |
| 42 | standing | falling down | 0.765 | 58 | 3 | non |
| 42 | short_label | falling down | 0.772 | 59 | 3 | non |
| 47 | standing | nausea/vomiting | 0.637 | 3 | 8 | non |
| 47 | seated | nausea/vomiting | 0.604 | 3 | 6 | non |
| 47 | short_label | nausea/vomiting | 0.461 | 58 | 11 | non |

**2/22 top-1 exact** (12 classes candidates). Classes 10/11/19/26 (reading,
writing, hat, jump up) exclues de ce run — unseen pour ss=5 seulement, sans
centroïde ss=12 disponible, à traiter séparément.

## Interprétation

Le signal n'est **pas dégénéré** : contrairement au tout premier essai (18/22
prédictions convergeant sur une seule classe, symptôme clair d'un bug de
pipeline), les prédictions se dispersent maintenant selon le contenu, avec des
confusions qui ont un sens physique :
- pick up ↔ falling down : flexion rapide du buste, dynamique proche
- falling down ↔ walking towards/apart : déplacement global du corps
- clapping systématiquement mal classé (rang 8-12) : cohérent avec la perte
  d'info main/doigts du retargeting — clapping est presque entièrement porté
  par le mouvement des mains

Signal encourageant : la variante **"seated"** l'emporte plus souvent que
"standing_or_neutral" sur les classes où c'est physiquement le bon contexte
(brush hair, tear up paper) — suggère un effet réel et mesurable du phrasé du
prompt, pas du bruit.

## Limites connues

- Squelettes 22-joints (HumanML3D) et 25-joints (NTU) sans détail doigts des
  deux côtés — désavantage structurel pour les classes dominées par le geste fin
  (clapping, et probablement reading/writing à tester).
- Recentrage sur la racine, pas de calibration caméra réelle — approximation
  suffisante ici (validée par le sanity-check sur données réelles) mais pas une
  reproduction exacte de la géométrie NTU/Kinect.
- Correspondance de joints dérivée manuellement, pas issue d'une validation
  publiée.
- Un seul run, 3 variantes de prompt par classe — pas de multi-seed, pas de
  test de stabilité.

## Verdict

Résultat suffisant pour justifier la suite du chantier : le pipeline
génère un signal réel, corrélé au contenu, avec des erreurs interprétables —
pas une preuve définitive, mais un GO raisonnable pour continuer plutôt qu'un
signal à l'arrêt.

## Prochaines étapes

1. Étendre aux 4 classes unseen ss=5-only (reading, writing, hat, jump up) avec
   le modèle ShiftGCN ss=5 correspondant.
2. Élargir le nombre de variantes de prompt par classe.
3. Diagnostic élargi : comparer aussi contre les centroïdes seen (pas seulement
   les 12 unseen) pour vérifier qu'un généré "unseen" ne dérive pas vers seen.
4. Si confirmé stable : intégration contrôlée dans l'entraînement du VAE
   (worktree dédié, une seule variable testée, comparé à H=52.61% de référence)
   — en gardant à l'esprit l'échec du précédent 5B (double extrapolation
   hors-distribution).

## Scripts (actuellement dans MotionGPT3/, hors dépôt git du projet)

- `compare_embeddings.py` — pipeline complet retargeting → ShiftGCN → comparaison
- `sanity_check_real.py` — validation de la reconstruction du modèle sur données réelles
- `visualize_ntu_skeleton.py` / `visualize_generated_skeleton.py` — rendu matplotlib
- `check_scale.py` — diagnostic d'échelle spatiale

Copiés dans ce worktree pour versioning (voir ci-dessous).
