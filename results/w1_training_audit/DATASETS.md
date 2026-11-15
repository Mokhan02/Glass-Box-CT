# W1 dataset records (made after the PREREG commit 9e1a515, 2026-10-12)

Data live under `data/w1/` (git-ignored, never committed). Downloaded files are untrusted: each in its own directory, read with `python -I`.

| Dataset | Licence (as read) | Source of the record | Cases | Size | Result vs PREREG |
|---|---|---|---|---|---|
| MSD Task03 Liver | CC-BY-SA 4.0 (`dataset.json` `licence`, release 1.0 04/05/2018; also AWS Open Data registry) | `dataset.json` read from the first 3 MB of the tar | numTraining 131, numTest 70; labels 0 background, 1 liver, 2 cancer | `Task03_Liver.tar` 28.93 GB (S3 object `s3://msd-for-monai/Task03_Liver.tar`, anonymous HTTPS) | matches (131 labelled, label 1 = liver, 2 = tumour) |
| CRML-CT = TCIA Colorectal-Liver-Metastases (Simpson et al., Sci Data 11:172, 2024) | CC BY 4.0 (IDC index `license_short_name`; TCIA page) | IDC index query (`idc-index`) | 197 patients, 197 CT series, 421 SEG series | CT 9.08 GB + SEG 2.29 GB (IDC), 11.64 GB total download | matches (197); SEG series count (421) is more than one per patient, so tumour/liver segment parsing must be checked on the first cases |
| MSD Task08 HepaticVessel | CC-BY-SA 4.0 (`dataset.json`, release 1.1 14/08/2018) | S3 object `s3://msd-for-monai/Task08_HepaticVessel.tar`, anonymous HTTPS | 303 labelled training cases; labels 1 Vessel, 2 Tumour (no liver label); downloaded 2026-11-15 to a folder outside the repo | `Task08_HepaticVessel.tar` 9,347,254,784 bytes, SHA-256 `ee880799f12e3b6e1ef2f8645f6626c5b39de77a4f1eae6f496c25fbf306ba04` | downloaded; pseudo-liver analysis not run (CPU time, see `source_contrast/README.md`) |

Not yet checked: whether the NV-Segment-CT bundle publishes per-case training splits (PREREG expects not).
Task03 download (2026-10-16, on the GPU instance, because local transfer was too slow): `Task03_Liver.tar` 28,925,891,584 bytes (equal to the source object size); `tar -tf` exit 0, 672 entries; SHA-256 `4007d9db1acda850d57a6ceb2b3998b7a0d43f8ad5a3f740dc38bc0cb8b7a2c5` ([task03_verify.log](task03_verify.log)). Extracted into its own empty directory: 131 images and 131 labels (the tar's macOS `._*` files ignored).
CRML-CT SHA-256 is not recorded: it came from the IDC index as many DICOM files rather than one archive (197 patients, CT and SEG series, 11.64 GB).
Task08: not downloaded (secondary analysis not run).
