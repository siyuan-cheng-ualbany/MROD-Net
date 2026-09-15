# MROD-Net

Code for *MROD-Net: Predicting Inter-Reader Disagreement and Ordinal Malignancy Ratings for Pulmonary Nodules on CT*.

The pipeline runs end to end from the public LIDC-IDRI archives and reproduces the dataset, the cross-validation folds and every number in the paper. Preprocessing takes about four minutes on 16 CPU cores; the full experimental grid is 300 GPU runs of roughly three minutes each.

## What is here

| | |
|---|---|
| `src/step1_parse_xml.py` | LIDC XML → reader-level nodule marks |
| `src/step2_series_meta.py` | DICOM headers → per-series geometry (needed to convert pixels to millimetres) |
| `src/step3_cluster.py` | marks → physical nodules, mutual-consistency grouping |
| `src/step4_patches.py` | fixed 64 mm field of view, 7 slices at 2 mm offsets |
| `src/step5_folds.py` | patient-level stratified 5-fold with nested validation |
| `src/models.py` | encoders, aggregation, heads, and the configuration registry |
| `src/train.py` | trains one (configuration, fold, seed) and writes per-nodule held-out predictions |
| `data/` | derived tables, small enough to ship: the 927-nodule study table with fold assignment, the series geometry, and the grouping summaries |

Image data is not included. It is public.

## Getting the data

```
# annotations, ~9 MB
curl -L -o lidc_xml.zip \
  https://wiki.cancerimagingarchive.net/download/attachments/1966254/LIDC-XML-only.zip

# CT series, 1018 zips, ~67 GB — see the TCIA REST API
#   https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?Collection=LIDC-IDRI&Modality=CT
#   https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=<uid>
```

## Reproducing

```
python src/step1_parse_xml.py --xml-zip lidc_xml.zip --out reader_nodules.csv --stats-out stage1_stats.csv
python src/step2_series_meta.py --dicom-dir dicom --out-series series_meta.csv --out-slices slice_index.csv
python src/step3_cluster.py --reader-nodules reader_nodules.csv --series-meta series_meta.csv \
    --out clusters.csv --summary-out cluster_summary.csv --xy-tol-mm 8 --no-transitive
python src/step4_patches.py --clusters clusters.csv --series-meta series_meta.csv \
    --out-dir patches --out-index patch_index.csv --fov-mm 64 --out-px 128 --n-slices 7 --z-step-mm 2.0
python src/step5_folds.py --patch-index patch_index.csv --out table.csv
python src/train.py --table table.csv --model mrou_no_lscore --fold 0 --seed 0 --out-dir runs
```

`--no-transitive` is the grouping rule used in the paper: a nodule group is formed only when every pair of marks inside it matches, rather than chaining through a third mark. Dropping the flag reproduces the transitive behaviour and the smaller 881-nodule dataset.

Run `train.py` over all configurations in `models.CONFIGS` × 5 folds × 3 seeds to reproduce the tables. Each run writes a metrics JSON and a per-nodule prediction CSV; the analysis reads those.

## Requirements

Python 3.11 with `torch`, `torchvision`, `timm` (Xception only), `pydicom`, `numpy`, `pandas`, `scipy`, `scikit-learn`.

## Notes

Two of the 300 runs stopped at or before epoch 3. A run whose best validation epoch is 3 or earlier was defined as a failed optimisation rather than an architectural result; both are reported in place rather than excluded.

## Citation

Cheng S., Yıldız O., Subasi A. *MROD-Net: Predicting Inter-Reader Disagreement and Ordinal Malignancy Ratings for Pulmonary Nodules on CT.* (under review)

## Contributors

Siyuan Cheng designed the multi-rater ordinal formulation and the network architecture, and implemented the first version of the pipeline. Osman Yıldız rewrote the pipeline in this repository, revised the annotation grouping and patch extraction procedures, and designed and ran the experiments.
