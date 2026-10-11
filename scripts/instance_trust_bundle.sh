#!/usr/bin/env bash
# One GPU session for the trust layer: (1) phase estimates for the 89-case cohort, (2) the flag-gated recalibration test on the saved logits,
# (3) rebuild the five demo scans so they carry the portal vein, the phase estimate and the new result fields, (4) pack the small outputs.
# Needs the persistent filesystem (CTGate) and the repo synced to ~/glassbox. Output: cache/trust_bundle.tgz
set -euo pipefail
cd ~/glassbox; V=/lambda/nfs/CTGate/glassbox/venv/bin/python
[ -e cache ] || ln -s /lambda/nfs/CTGate/glassbox/cache cache
[ -d cache/demo ] || cp -r /lambda/nfs/CTGate/glassbox/demo cache/demo
mkdir -p results/trust
$V scripts/gated_recal_run.py > results/trust/gated_recal.log 2>&1 &            # CPU, runs alongside the GPU job
$V scripts/phase_cohort.py > results/trust/phase_cohort.log 2>&1
wait
$V scripts/make_demo_results.py --stage rebuild
$V scripts/export_reference.py --demo cache/demo --out cache/demo_ref
tar -czf cache/trust_bundle.tgz results/trust/phase_estimates.csv results/trust/gated_recal.json results/trust/GATED_RECAL.md results/trust/lesion_measurements.csv results/trust/*.log -C cache demo demo_ref
ls -l cache/trust_bundle.tgz; echo TRUST-BUNDLE-READY
