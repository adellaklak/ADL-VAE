#!/bin/bash
cd "$(cd "$(dirname "$0")" && pwd)"
z() { grep -oP 'increased to \K[0-9.]+' "$1" 2>/dev/null | tail -1; }
echo "=== DESCRIPTIONS (seed 5, baseline lb_ad_md : 84.44 / 48.6) ==="
printf "%-18s %-10s %-10s\n" "tm" "ss=5" "ss=12"
for TM in lb ad md ad_md lb_ad_md lbac_md lbac_md_bdavg dual dual2; do
  printf "%-18s %-10s %-10s\n" "$TM" "$(z desc_${TM}_5.log)" "$(z desc_${TM}_12.log)"
done
