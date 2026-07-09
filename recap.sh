#!/bin/bash
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE_clean

zsl() { grep -oP 'increased to \K[0-9.]+' "$1" 2>/dev/null | tail -1; }
ok()  { [ -f "results/$1/ViT-B/32/lb_ad_md/se_16999.pth.tar" ] && echo "10c" || echo "--"; }
crash() { local f=$(ls -t OAR.*.stderr 2>/dev/null | head -20 | xargs grep -l "libtorch\|No CUDA" 2>/dev/null | wc -l); echo "$f"; }

echo "=========================================================="
echo " BASELINE (esterel7, seed=5, lb_ad_md, code officiel)"
echo "   ss=5  : ZSL 84.44   H_oracle 86.5   H_real 63.97"
echo "   ss=12 : ZSL 48.6    H_oracle 62.7   H_real 47.10"
echo "=========================================================="
echo ""

echo "### VARIANCE GPU — baseline ss=12 (LA mesure de reference) ###"
vals=""
for N in 5 7 10 11 19 23 29; do
  z=$(zsl var12_esterel$N.log)
  printf "   esterel%-3s ZSL=%-8s [%s]\n" "$N" "${z:---}" "$(ok var12_esterel$N)"
  [ -n "$z" ] && vals="$vals $z"
done
if [ -n "$vals" ]; then
  echo "$vals" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
    {a[NR]=$1; s+=$1}
    END{if(NR>0){m=s/NR; for(i=1;i<=NR;i++) v+=(a[i]-m)^2;
    printf "   -> n=%d  min=%.2f  max=%.2f  amplitude=%.2f  moy=%.2f  ecart-type=%.2f\n", NR,a[1],a[NR],a[NR]-a[1],m,sqrt(v/NR)}}'
  echo "   >>> tout gain ss=12 inferieur a l'amplitude ci-dessus n'est PAS un signal"
fi
echo ""

echo "### STRUCTURE — balayage align_w (GPU fixe) ###"
echo "   ss=12 (baseline 48.6) :"
for AW in 0.5 1.0 2.0 4.0; do
  printf "     w=%-4s ZSL=%-8s [%s]\n" "$AW" "$(zsl structure_12_w${AW}.log)" "$(ok structure_12_w${AW})"
done
echo "   ss=5 (baseline 84.44) :"
for AW in 0.5 1.0 2.0; do
  printf "     w=%-4s ZSL=%-8s [%s]\n" "$AW" "$(zsl structure_5_w${AW}.log)" "$(ok structure_5_w${AW})"
done
echo ""

echo "### CONFUSION — ablation (d'ou vient le gain ?) ###"
echo "   attendu : ema_nograd == none  (gradient nul sur le terme de confusion)"
for SS in 5 12; do
  base=$([ $SS = 5 ] && echo 84.44 || echo 48.6)
  echo "   ss=$SS (baseline $base) :"
  for IMPL in none ema_nograd sample_grad; do
    printf "     hinge_asym + %-12s ZSL=%-8s [%s]\n" "$IMPL" \
      "$(zsl conf_hinge_asym_${IMPL}_${SS}.log)" "$(ok conf_hinge_asym_${IMPL}_${SS}_r)"
  done
done
echo ""

echo "### CONFIG PAPIER (alpha=0.1 / nepc=1900) ###"
echo "   Table 12 du papier : alpha=0.1, epochs=1900, phi=35  |  script officiel : alpha=0.5, 1700, phi=30"
for TAG in a01 e1900; do
  for SS in 5 12; do
    printf "   %-6s ss=%-3s ZSL=%-8s [%s]\n" "$TAG" "$SS" "$(zsl cfg_${TAG}_${SS}.log)" "$(ok cfg_${TAG}_${SS}_r)"
  done
done
echo ""

echo "### JOBS ENCORE ACTIFS ###"
oarstat -u $USER 2>/dev/null | tail -n +3
echo ""
echo "### CRASHS (libtorch / No CUDA) parmi les 20 derniers stderr ###"
ls -t OAR.*.stderr 2>/dev/null | head -20 | while read f; do
  grep -lq "libtorch\|No CUDA" "$f" 2>/dev/null && echo "   $f : $(tail -1 $f | cut -c1-70)"
done
echo "   (rien ci-dessus = aucun crash)"
