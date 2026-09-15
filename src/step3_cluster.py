"""Stage 3: group reader-level annotations into physical nodules.

Two differences from the original pipeline, both reported in the summary so the
effect of each is auditable:

  1. In-plane distances are converted to MILLIMETRES using the series pixel
     spacing before thresholding. The original compared a pixel distance against
     a fixed "12" while comparing z in mm, so the physical tolerance silently
     varied by more than 2x across scans.
  2. Transitive closure (union-find) is optional. With --no-transitive a group is
     only formed when every pair inside it matches, which prevents A-B and B-C
     merging A and C when A and C do not match each other.
"""
import argparse
import numpy as np
import pandas as pd


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_series(g, xy_tol_mm, z_tol_mm, size_frac, transitive, stats):
    """Return a list of index-lists, one per physical nodule."""
    n = len(g)
    pairs = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(pairs, True)

    for i in range(n):
        a = g.iloc[i]
        for j in range(i + 1, n):
            b = g.iloc[j]
            stats["pairs_compared"] += 1
            if a.reader_idx == b.reader_idx:
                stats["pairs_rejected_same_reader"] += 1
                continue
            z_overlap = not (a.max_z_mm < b.min_z_mm or b.max_z_mm < a.min_z_mm)
            z_close = abs(a.center_z_mm - b.center_z_mm) <= z_tol_mm
            dxy = float(np.hypot(a.center_x_mm - b.center_x_mm, a.center_y_mm - b.center_y_mm))
            size_ref = max(a.diam_xy_mm, b.diam_xy_mm)
            thr = max(xy_tol_mm, size_frac * size_ref)
            if (z_overlap or z_close) and dxy <= thr:
                pairs[i, j] = pairs[j, i] = True
                stats["pairs_matched"] += 1
            else:
                stats["pairs_rejected_spatial"] += 1

    if transitive:
        d = DSU(n)
        for i in range(n):
            for j in range(i + 1, n):
                if pairs[i, j]:
                    d.union(i, j)
        groups = {}
        for i in range(n):
            groups.setdefault(d.find(i), []).append(i)
        out = list(groups.values())
        for grp in out:
            if len(grp) > 1:
                sub = pairs[np.ix_(grp, grp)]
                if not sub.all():
                    stats["clusters_not_fully_connected"] += 1
        return out

    # greedy maximal cliques: seed on the densest unassigned node
    unassigned = set(range(n))
    out = []
    while unassigned:
        seed = max(unassigned, key=lambda i: pairs[i, list(unassigned)].sum())
        grp = [seed]
        unassigned.discard(seed)
        cands = sorted([c for c in unassigned if pairs[seed, c]],
                       key=lambda c: -pairs[c, list(unassigned)].sum())
        for c in cands:
            if all(pairs[c, m] for m in grp):
                grp.append(c)
                unassigned.discard(c)
        out.append(grp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader-nodules", required=True)
    ap.add_argument("--series-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--xy-tol-mm", type=float, default=6.0,
                    help="floor on the in-plane matching distance, in mm")
    ap.add_argument("--z-tol-mm", type=float, default=3.0)
    ap.add_argument("--size-frac", type=float, default=0.6,
                    help="matching distance also scales with nodule diameter")
    ap.add_argument("--no-transitive", action="store_true",
                    help="require every pair in a cluster to match")
    args = ap.parse_args()

    rd = pd.read_csv(args.reader_nodules)
    sm = pd.read_csv(args.series_meta)
    sm = sm[sm.error.fillna("") == ""][
        ["series_uid", "patient_id", "row_spacing_mm", "col_spacing_mm", "thickness_mm", "z_step_mm", "n_slices"]]

    before = len(rd)
    rd = rd.merge(sm, on="series_uid", how="inner")
    print(f"reader rows: {before} -> {len(rd)} after joining series geometry "
          f"({before - len(rd)} had no matching CT series)")

    # pixels -> millimetres
    rd["center_x_mm"] = rd.center_x_px * rd.col_spacing_mm
    rd["center_y_mm"] = rd.center_y_px * rd.row_spacing_mm
    rd["diam_xy_mm"] = np.maximum(rd.diam_x_px * rd.col_spacing_mm,
                                  rd.diam_y_px * rd.row_spacing_mm)

    stats = dict(pairs_compared=0, pairs_rejected_same_reader=0, pairs_matched=0,
                 pairs_rejected_spatial=0, clusters_not_fully_connected=0)
    rows = []
    for series_uid, g in rd.groupby("series_uid"):
        g = g.reset_index(drop=True)
        for k, idxs in enumerate(cluster_series(g, args.xy_tol_mm, args.z_tol_mm,
                                                args.size_frac, not args.no_transitive, stats)):
            sub = g.iloc[idxs]
            readers = sorted(sub.reader_idx.unique().tolist())
            mal = sorted(sub.malignancy.tolist())
            by_reader = sub.sort_values("reader_idx").malignancy.tolist()
            rows.append({
                "cluster_id": f"{series_uid}__{k}",
                "series_uid": series_uid,
                "study_uid": sub.study_uid.iat[0],
                "patient_id": sub.patient_id.iat[0],
                "n_marks": len(sub),
                "n_unique_readers": len(readers),
                "reader_list": str(readers),
                "malignancy_list": str(mal),
                "malignancy_by_reader": str(by_reader),
                "score_mean": float(np.mean(mal)),
                "score_var_pop": float(np.var(mal)),
                "center_x_mm": float(sub.center_x_mm.mean()),
                "center_y_mm": float(sub.center_y_mm.mean()),
                "center_z_mm": float(sub.center_z_mm.mean()),
                "center_x_px": float(sub.center_x_px.mean()),
                "center_y_px": float(sub.center_y_px.mean()),
                "diam_xy_mm": float(sub.diam_xy_mm.mean()),
                "z_span_mm": float(sub.max_z_mm.max() - sub.min_z_mm.min()),
                "row_spacing_mm": float(sub.row_spacing_mm.iat[0]),
                "col_spacing_mm": float(sub.col_spacing_mm.iat[0]),
                "thickness_mm": float(sub.thickness_mm.iat[0]),
                "z_step_mm": float(sub.z_step_mm.iat[0]),
            })

    cdf = pd.DataFrame(rows)
    cdf.to_csv(args.out, index=False)

    dist = cdf.n_unique_readers.value_counts().sort_index()
    summary = [{"category": k, "count": v} for k, v in stats.items()]
    summary += [{"category": "clusters_created", "count": len(cdf)},
                {"category": "series_processed", "count": int(cdf.series_uid.nunique())},
                {"category": "patients", "count": int(cdf.patient_id.nunique())}]
    for r, c in dist.items():
        summary.append({"category": f"clusters_with_{r}_readers", "count": int(c)})
    four = cdf[(cdf.n_unique_readers == 4) & (cdf.n_marks == 4)]
    summary.append({"category": "clusters_4_readers_exactly_4_marks", "count": len(four)})
    pd.DataFrame(summary).to_csv(args.summary_out, index=False)

    print("\n--- stage 3 summary ---")
    for s in summary:
        print(f"  {s['category']:38s} {s['count']}")
    print(f"\n  in-plane tolerance: max({args.xy_tol_mm} mm, {args.size_frac} x diameter)"
          f"   z tolerance: overlap or <= {args.z_tol_mm} mm")
    print(f"  clustering mode: {'transitive closure' if not args.no_transitive else 'fully connected groups'}")
    print(f"\n  usable 4-reader nodules: {len(four)}")


if __name__ == "__main__":
    main()
