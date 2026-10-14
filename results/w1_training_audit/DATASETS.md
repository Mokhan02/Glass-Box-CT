# W1 dataset records (made after the PREREG commit 9e1a515, 2026-10-12)

Data live under `data/w1/` (git-ignored, never committed). Downloaded files are untrusted: each in its own directory, read with `python -I`.

| Dataset | Licence (as read) | Source of the record | Cases | Size | Result vs PREREG |
|---|---|---|---|---|---|
| MSD Task03 Liver | CC-BY-SA 4.0 (`dataset.json` `licence`, release 1.0 04/05/2018; also AWS Open Data registry) | `dataset.json` read from the first 3 MB of the tar | numTraining 131, numTest 70; labels 0 background, 1 liver, 2 cancer | `Task03_Liver.tar` 28.93 GB (S3 object `s3://msd-for-monai/Task03_Liver.tar`, anonymous HTTPS) | matches (131 labelled, label 1 = liver, 2 = tumour) |
| CRML-CT = TCIA Colorectal-Liver-Metastases (Simpson et al., Sci Data 11:172, 2024) | CC BY 4.0 (IDC index `license_short_name`; TCIA page) | IDC index query (`idc-index`) | 197 patients, 197 CT series, 421 SEG series | CT 9.08 GB + SEG 2.29 GB (IDC), 11.64 GB total download | matches (197); SEG series count (421) is more than one per patient, so tumour/liver segment parsing must be checked on the first cases |
| MSD Task08 HepaticVessel | as MSD | S3 listing | not yet downloaded (secondary) | `Task08_HepaticVessel.tar` 9.35 GB | pending |

Not yet checked: whether the NV-Segment-CT bundle publishes per-case training splits (PREREG expects not).
Sha256 of the downloaded files will be appended below when the downloads finish.
