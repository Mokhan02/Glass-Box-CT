# MSD Task03: the 16 cases that did not enter (diagnostic only, not used in any analysis)

Included cases: 115; tumour voxels, median 15544, 5th-95th percentile 576 to 817945.

- 13 tumour-free cases (no label 2): liver_105, liver_106, liver_114, liver_115, liver_119, liver_32, liver_34, liver_38, liver_41, liver_47, liver_87, liver_89, liver_91. They contain no tumour, so their exclusion cannot depend on tumour size or brightness (no tumour exists to measure).

- liver_39: failed the surface guard (tumour surface within 3 mm of liver label: 0.24; tumour/liver volume 0.117). Tumour voxels 348986 (percentile among included cases: 88); lesions >= 10 mm: 1; contrast (HU, diameter mm): -8.0 (116).
- liver_71: failed the surface guard (tumour surface within 3 mm of liver label: 0.40; tumour/liver volume 0.153). Tumour voxels 142741 (percentile among included cases: 77); lesions >= 10 mm: 3; contrast (HU, diameter mm): -20.7 (96), -24.9 (25), -15.1 (25).
- liver_74: failed the surface guard (tumour surface within 3 mm of liver label: 0.40; tumour/liver volume 0.052). Tumour voxels 37144 (percentile among included cases: 61); lesions >= 10 mm: 2; contrast (HU, diameter mm): -17.8 (77), -49.5 (14).

Included cases that contributed no lesion (no tumour component >= 10 mm): liver_112, liver_83; this is the pre-registered 10 mm lesion rule, which is size-related by definition.
