"""Stage 5: patient-level stratified 5-fold assignment, plus a nested
patient-level validation split inside each training fold.

Both splits are made once, saved, and reused by every model so comparisons are
paired. Nothing here touches the image data.
"""
import argparse, ast
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def score_bin(x):
    if x < 2.0:
        return "low"
    if x < 3.0:
        return "low_mid"
    if x < 4.0:
        return "high_mid"
    return "high"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.125)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.patch_index)
    df = df[df.patch_path.astype(str).str.len() > 0].copy()

    # sorted reader scores + the targets derived from them
    mal = df.malignancy_list.apply(lambda s: sorted(ast.literal_eval(s)))
    for k in range(4):
        df[f"score_{k+1}"] = mal.apply(lambda v: v[k])
    # ratings in reading-session order, for the unsorted-target baseline
    raw = df.malignancy_by_reader.apply(ast.literal_eval)
    for k in range(4):
        df[f"raw_score_{k+1}"] = raw.apply(lambda v: v[k])
    df["score_mean"] = mal.apply(np.mean)
    df["score_var_pop"] = mal.apply(lambda v: float(np.var(v)))
    df["score_var_norm"] = df.score_var_pop / 4.0
    df["score_std_norm"] = np.sqrt(df.score_var_pop) / 2.0

    pat = df.groupby("patient_id").agg(mean_score=("score_mean", "mean"),
                                       n_nodules=("cluster_id", "size")).reset_index()
    pat["score_bin"] = pat.mean_score.apply(score_bin)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pat["fold"] = -1
    for f, (_, te) in enumerate(skf.split(pat, pat.score_bin)):
        pat.loc[te, "fold"] = f

    # nested validation patients, one assignment per fold, reused by every model
    for f in range(args.n_folds):
        tv = pat[pat.fold != f]
        _, va = train_test_split(tv, test_size=args.val_frac,
                                 random_state=args.seed + f, stratify=tv.score_bin)
        pat[f"is_val_fold{f}"] = pat.patient_id.isin(va.patient_id)

    cols = ["patient_id", "fold"] + [f"is_val_fold{f}" for f in range(args.n_folds)]
    df = df.merge(pat[cols], on="patient_id", how="left")
    df.to_csv(args.out, index=False)

    print("--- stage 5 summary ---")
    print(f"  nodules {len(df)}   patients {df.patient_id.nunique()}   series {df.series_uid.nunique()}")
    print("\n  fold sizes (nodules / patients):")
    for f in range(args.n_folds):
        s = df[df.fold == f]
        tr = df[(df.fold != f) & (~df[f"is_val_fold{f}"])]
        va = df[(df.fold != f) & (df[f"is_val_fold{f}"])]
        print(f"    fold {f}: test {len(s):4d}/{s.patient_id.nunique():3d}   "
              f"train {len(tr):4d}/{tr.patient_id.nunique():3d}   val {len(va):3d}/{va.patient_id.nunique():3d}")

    print("\n  leak check, shared patients between folds:")
    bad = 0
    for i in range(args.n_folds):
        for j in range(i + 1, args.n_folds):
            n = len(set(df[df.fold == i].patient_id) & set(df[df.fold == j].patient_id))
            bad += n
            if n:
                print(f"    fold {i} n fold {j}: {n}")
    print(f"    total overlapping patients: {bad}")
    for f in range(args.n_folds):
        tr = set(df[(df.fold != f) & (~df[f"is_val_fold{f}"])].patient_id)
        va = set(df[(df.fold != f) & (df[f"is_val_fold{f}"])].patient_id)
        te = set(df[df.fold == f].patient_id)
        assert not (tr & va) and not (tr & te) and not (va & te), f"leak in fold {f}"
    print("    train/val/test patient sets disjoint in every fold: OK")

    print("\n  score_mean by fold:", {f: round(df[df.fold == f].score_mean.mean(), 3)
                                     for f in range(args.n_folds)})
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
