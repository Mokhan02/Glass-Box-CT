# glassbox-ct-demo-data

Five resampled research CT scans and their pre-computed results, used by the [Glass-Box CT](https://github.com/Mokhan02/Glass-Box-CT) demo. **Research use only. Not for clinical decisions.**

The data is published as a release asset: `glassbox-demo-data-v1.tar.gz` (56402560 bytes, SHA-256 `8391cd183116664dad64a79d48d9e2996ce5d4750a8219dca76e54f9d321cdb5`). The Glass-Box CT Docker build downloads it and refuses to continue if the checksum differs.

Licence and attribution: the scans are from HCC-TACE-Seg on The Cancer Imaging Archive, **CC BY 4.0**. They were resampled for viewing; the segmentations are NVIDIA NV-Segment-CT (VISTA-3D) output; the reference outline is the dataset's own annotation. The full citation, DOI and a statement of changes are in `demo/ATTRIBUTION.md` inside the archive and are reproduced here:

- Moawad, A. W. et al. (2021). *Multimodality annotated HCC cases with and without advanced imaging segmentation (HCC-TACE-Seg)* [Data set]. TCIA. https://doi.org/10.7937/TCIA.5FNA-0924
- Moawad, A. W. et al. (2023). Multimodality annotated hepatocellular carcinoma data set including pre- and post-TACE with imaging segmentation. *Scientific Data* 10, 33. https://doi.org/10.1038/s41597-023-01928-3
- Clark, K. et al. (2013). The Cancer Imaging Archive (TCIA). *Journal of Digital Imaging* 26(6), 1045-1057. https://doi.org/10.1007/s10278-013-9622-7

Verify a download: `shasum -a 256 glassbox-demo-data-v1.tar.gz` must print `8391cd183116664dad64a79d48d9e2996ce5d4750a8219dca76e54f9d321cdb5`.
