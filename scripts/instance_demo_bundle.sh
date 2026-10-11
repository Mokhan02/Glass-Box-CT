#!/usr/bin/env bash
# One GPU/storage session to finish the five demo scans. Needs the persistent filesystem (CTGate) mounted and the repo synced to ~/glassbox.
# 1. regenerate the demo results from the ORIGINAL CTs with the recalibrated (research) threshold and the viewer-grid alignment,
# 2. export the dataset's reference tumour annotation for each demo scan,
# 3. pack everything small enough to copy back (no original CTs leave the instance).
set -euo pipefail
cd ~/glassbox; V=/lambda/nfs/CTGate/glassbox/venv/bin/python
[ -e cache ] || ln -s /lambda/nfs/CTGate/glassbox/cache cache
[ -d cache/demo ] || cp -r /lambda/nfs/CTGate/glassbox/demo cache/demo
$V scripts/make_demo_results.py --stage rebuild
$V scripts/export_reference.py --demo cache/demo --out cache/demo_ref
tar -C cache -czf cache/demo_bundle.tgz demo demo_ref
ls -l cache/demo_bundle.tgz; echo BUNDLE-READY
