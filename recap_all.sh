#!/bin/bash
cd /srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/FSVAE_clean
z() { grep -oP 'increased to \K[0-9.]+' "$1" 2>/dev/null | tail -1; }

echo "=============== BASELINE (esterel7, seed 5) ==============="
echo "  ss=5  ZSL 84.44   |  ss=12  ZSL 48.6"
echo "  bruit GPU ss=12 : amplitude 0.52, sigma 0.21 (n=5)"
echo "  bruit GPU ss=5  : amplitude 0.37 (n=8)"
echo ""
echo "=============== VARIANCE GPU baseline ss=12 ==============="
for N in 5 7 10 11 19 23 29; do printf "  esterel%-3s %s\n" "$N" "$(z var12_esterel$N.log)"; done
echo ""
echo "=============== STRUCTURE ==============="
echo "  ss=12 (base 48.6) :"
for AW in 0.5 1.0 2.0 4.0; do printf "    w=%-4s %s\n" "$AW" "$(z structure_12_w${AW}.log)"; done
echo "  ss=5 (base 84.44) :"
for AW in 0.5 1.0 2.0; do printf "    w=%-4s %s\n" "$AW" "$(z structure_5_w${AW}.log)"; done
echo ""
echo "=============== CONFUSION (base loss = hinge_asym) ==============="
for SS in 5 12; do
  echo "  ss=$SS :"
  for I in none ema_nograd sample_grad; do printf "    %-12s %s\n" "$I" "$(z conf_hinge_asym_${I}_${SS}.log)"; done
done
echo ""
echo "=============== HINGE (train_hinge.py) ==============="
for SS in 12 5; do for PW in 0 1; do for M in 1.0 2.0 4.0; do
  v=$(z hinge_pw${PW}_m${M}_${SS}.log); [ -n "$v" ] && printf "  ss=%-3s pw=%s m=%-4s %s\n" "$SS" "$PW" "$M" "$v"
done; done; done
[ -z "$(ls hinge_*.log 2>/dev/null)" ] && echo "  AUCUN LOG -> train_hinge.py absent, jobs crashes"
echo ""
echo "=============== CONFIG (alpha / epochs) ==============="
for T in a01 e1900; do for SS in 5 12; do printf "  %-6s ss=%-3s %s\n" "$T" "$SS" "$(z cfg_${T}_${SS}.log)"; done; done
echo ""
echo "=============== MAHALANOBIS (eval sur runs existants) ==============="
for f in maha_*.log; do [ -f "$f" ] && { echo "  --- $f ---"; grep -E "ZSL=|gain" "$f" | sed 's/^/    /'; }; done
[ -z "$(ls maha_*.log 2>/dev/null)" ] && echo "  pas encore de resultats"
echo ""
echo "=============== JOBS ACTIFS ==============="
oarstat -u $USER 2>/dev/null | tail -n +3
echo ""
echo "=============== CRASHS RECENTS ==============="
ls -t OAR.*.stderr 2>/dev/null | head -15 | while read f; do
  e=$(grep -m1 -E "libtorch|No CUDA|No such file|not found|Error" "$f" 2>/dev/null | cut -c1-65)
  [ -n "$e" ] && echo "  $f : $e"
done
