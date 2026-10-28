# MROD-Net

Code for *MROD-Net: Predicting Inter-Reader Disagreement and Ordinal Malignancy Ratings for Pulmonary Nodules on CT*.

The pipeline runs end to end from the public LIDC-IDRI archives and reproduces the dataset, the cross-validation folds and every number in the paper. Preprocessing takes about four minutes on 16 CPU cores; the full experimental grid is 375 GPU runs of roughly three minutes each (300 in the main grid, 75 for the projection-width and slice-count experiments).

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
| `data/` | derived tables, small enough to ship: the 927-nodule study table with fold assignment, the series geometry, the grouping summaries, and the held-out predictions of every reported run |

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

Run `train.py` over all configurations in `models.CONFIGS` × 5 folds × 3 seeds to reproduce the tables; `proj256` and `proj1024` are the projection-width comparison. For the slice-count comparison, rebuild the patches with `--n-slices 3`, `5` or `9` (same `--z-step-mm 2.0`), build the table with `step5_folds.py` (the folds are the same), and train `mrou_no_lscore` with the matching `--n-slices`:

```
python src/train.py --table table_n9.csv --model mrou_no_lscore --n-slices 9 --fold 0 --seed 0 --out-dir runs_ns
```

Each run writes a metrics JSON and a per-nodule prediction CSV; the analysis reads those.

## Held-out predictions

`data/oof_predictions.csv.gz` holds the per-nodule held-out predictions of every run reported in the paper: 375 runs, 69,525 rows.

- the main grid: the 20 configurations used in the result tables × 5 folds × 3 seeds (300 runs);
- the projection width: `proj256` and `proj1024` (30 runs; 512 is the main grid's `mrou_no_lscore`);
- the number of slices: `mrou_no_lscore` with 3, 5 and 9 slices (45 runs; 7 is the main grid).

`mrou_no_lscore` is the proposed model, MROD-Net. Each combination of `model`, `n_slices` and `seed` covers all 927 nodules exactly once, each predicted by the fold in which it was held out.

| column | |
|---|---|
| `cluster_id`, `patient_id` | nodule and patient; join to `data/table_strict.csv` |
| `model`, `n_slices`, `fold`, `seed` | the run |
| `true_mean`, `true_var_norm` | observed mean rating and normalised rating variance |
| `pred_mean`, `pred_var_norm` | predicted mean rating and predicted disagreement |
| `pred_score_1` … `pred_score_4` | the four predicted ratings, sorted |
| `true_score_1` … `true_score_4` | the four observed ratings, sorted |

## Requirements

Python 3.11 with `torch`, `torchvision`, `timm` (Xception only), `pydicom`, `numpy`, `pandas`, `scipy`, `scikit-learn`.

## Notes

Three runs stopped at or before epoch 3: two in the main grid (`nols_no_pos` fold 1 seed 0, `single_ordinal_attn` fold 2 seed 2) and one at 256-dimensional projection (`proj256` fold 0 seed 1). A run whose best validation epoch is 3 or earlier was defined as a failed optimisation rather than an architectural result; all three are reported in place rather than excluded, and their predictions are in the file above.

## Citation

Cheng S., Yıldız O., Subasi A. *MROD-Net: Predicting Inter-Reader Disagreement and Ordinal Malignancy Ratings for Pulmonary Nodules on CT.* (under review)

## Contributors

Siyuan Cheng designed the multi-rater ordinal formulation and the network architecture, and implemented the first version of the pipeline. Osman Yıldız rewrote the pipeline in this repository, revised the annotation grouping and patch extraction procedures, and designed and ran the experiments.
