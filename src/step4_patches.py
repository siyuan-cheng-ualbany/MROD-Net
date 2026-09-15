"""Stage 4: extract a fixed-physical-size patch stack per nodule.

Three differences from the original pipeline:

  1. Fixed field of view in MILLIMETRES, resampled to a fixed pixel grid. The
     original cropped 128 native pixels, so the anatomy covered varied from
     about 59 mm to 125 mm depending on the scan.
  2. Slices are chosen at fixed physical offsets from the nodule centre, not by
     array index. Seven native slices spanned anywhere from 4 mm to 35 mm of
     anatomy because LIDC slice thickness ranges from 0.6 mm to 5 mm.
  3. Requests that fall outside the volume are PADDED with air (-1000 HU) and
     flagged, instead of silently clamping to a neighbouring slice. Clamping
     moved the nodule away from the centre index while the model kept querying
     the middle slice as if the nodule were still there.
"""
import argparse, io, os, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import pydicom

HU_MIN, HU_MAX, HU_AIR = -1000.0, 400.0, -1000.0


def load_volume(zip_path):
    """Return (volume HWZ-sorted list, z array, spacing, origin) for one series."""
    sl = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".dcm"):
                continue
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(name)), force=True)
            except Exception:
                continue
            if getattr(ds, "Modality", None) != "CT" or not hasattr(ds, "ImagePositionPatient"):
                continue
            try:
                arr = ds.pixel_array.astype(np.float32)
            except Exception:
                continue
            arr = arr * float(getattr(ds, "RescaleSlope", 1.0) or 1.0) \
                      + float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            sl.append((float(ds.ImagePositionPatient[2]), arr,
                       float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])))
    if not sl:
        return None
    sl.sort(key=lambda t: t[0])
    z = np.array([t[0] for t in sl], dtype=np.float64)
    vol = [t[1] for t in sl]
    return vol, z, (sl[0][2], sl[0][3])


def resample_plane(img, cx_px, cy_px, row_sp, col_sp, fov_mm, out_px):
    """Bilinear resample a square FOV (mm) centred on a pixel coordinate."""
    half = fov_mm / 2.0
    # target sample positions in mm relative to centre, then back to source pixels
    t = np.linspace(-half, half, out_px, dtype=np.float32)
    xs = cx_px + t / col_sp
    ys = cy_px + t / row_sp
    H, W = img.shape
    x0 = np.floor(xs).astype(np.int64); y0 = np.floor(ys).astype(np.int64)
    wx = (xs - x0).astype(np.float32);  wy = (ys - y0).astype(np.float32)
    x1, y1 = x0 + 1, y0 + 1
    valid_x = (x0 >= 0) & (x1 < W)
    valid_y = (y0 >= 0) & (y1 < H)
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    Ia = img[np.ix_(y0c, x0c)]; Ib = img[np.ix_(y0c, x1c)]
    Ic = img[np.ix_(y1c, x0c)]; Id = img[np.ix_(y1c, x1c)]
    wxg = wx[None, :]; wyg = wy[:, None]
    out = (Ia * (1 - wxg) * (1 - wyg) + Ib * wxg * (1 - wyg)
           + Ic * (1 - wxg) * wyg + Id * wxg * wyg)
    out[~valid_y[:, None] | ~valid_x[None, :]] = HU_AIR
    return out.astype(np.float32), int((~valid_x).sum() + (~valid_y).sum())


def process_series(zip_path, nods, out_dir, fov_mm, out_px, n_slices, z_step_mm):
    loaded = load_volume(zip_path)
    if loaded is None:
        return [{"cluster_id": n["cluster_id"], "error": "no_volume"} for n in nods]
    vol, zarr, (row_sp, col_sp) = loaded
    half = n_slices // 2
    recs = []
    for n in nods:
        targets = n["center_z_mm"] + np.arange(-half, half + 1) * z_step_mm
        planes, n_out, n_clip = [], 0, 0
        for tz in targets:
            k = int(np.argmin(np.abs(zarr - tz)))
            if abs(zarr[k] - tz) > z_step_mm:       # no slice near this offset
                planes.append(np.full((out_px, out_px), HU_AIR, dtype=np.float32))
                n_out += 1
                continue
            p, clipped = resample_plane(vol[k], n["center_x_px"], n["center_y_px"],
                                        row_sp, col_sp, fov_mm, out_px)
            planes.append(p)
            n_clip += clipped
        stack = np.stack(planes, 0)
        stack = np.clip(stack, HU_MIN, HU_MAX)
        stack = (stack - HU_MIN) / (HU_MAX - HU_MIN)
        fn = n["cluster_id"].replace("/", "_") + ".npy"
        np.save(os.path.join(out_dir, fn), stack.astype(np.float32))
        recs.append({
            "cluster_id": n["cluster_id"], "patch_file": fn,
            "n_slices_padded": n_out, "n_inplane_clipped": n_clip,
            "edge_case": bool(n_out > 0 or n_clip > 0),
            "z_span_covered_mm": float(n_slices * z_step_mm),
            "fov_mm": fov_mm, "error": "",
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--series-meta", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--fov-mm", type=float, default=64.0)
    ap.add_argument("--out-px", type=int, default=128)
    ap.add_argument("--n-slices", type=int, default=7)
    ap.add_argument("--z-step-mm", type=float, default=2.0)
    ap.add_argument("--min-readers", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cdf = pd.read_csv(args.clusters)
    keep = cdf[(cdf.n_unique_readers >= args.min_readers) & (cdf.n_marks == cdf.n_unique_readers)].copy()
    print(f"clusters: {len(cdf)} -> {len(keep)} with >= {args.min_readers} readers and one mark each")

    sm = pd.read_csv(args.series_meta)
    zip_of = dict(zip(sm.series_uid, sm.zip_path))
    keep["zip_path"] = keep.series_uid.map(zip_of)
    missing = keep.zip_path.isna().sum()
    if missing:
        print(f"  dropping {missing} clusters with no DICOM zip")
        keep = keep.dropna(subset=["zip_path"])

    groups = {}
    for _, r in keep.iterrows():
        groups.setdefault(r.zip_path, []).append(
            {"cluster_id": r.cluster_id, "center_z_mm": r.center_z_mm,
             "center_x_px": r.center_x_px, "center_y_px": r.center_y_px})

    recs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_series, zp, nods, args.out_dir,
                          args.fov_mm, args.out_px, args.n_slices, args.z_step_mm): zp
                for zp, nods in groups.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            recs.extend(fut.result())
            if i % 50 == 0:
                print(f"  {i}/{len(groups)} series", flush=True)

    rdf = pd.DataFrame(recs)
    out = keep.merge(rdf, on="cluster_id", how="left")
    out["patch_path"] = out.patch_file.apply(
        lambda f: os.path.join(args.out_dir, f) if isinstance(f, str) else "")
    out.to_csv(args.out_index, index=False)

    ok = out[out.error.fillna("") == ""]
    print("\n--- stage 4 summary ---")
    print(f"  patches written         {len(ok)}")
    print(f"  unique patients         {ok.patient_id.nunique()}")
    print(f"  field of view           {args.fov_mm} mm -> {args.out_px} px "
          f"({args.fov_mm/args.out_px:.3f} mm/px)")
    print(f"  z coverage              {args.n_slices} slices x {args.z_step_mm} mm "
          f"= {args.n_slices*args.z_step_mm} mm")
    print(f"  needed z padding        {int((ok.n_slices_padded > 0).sum())} nodules")
    print(f"  touched the image edge  {int((ok.n_inplane_clipped > 0).sum())} nodules")
    print(f"  any edge effect         {int(ok.edge_case.sum())} nodules "
          f"<-- silently mishandled by the original pipeline")
    print("\n  score_mean distribution:")
    print(ok.score_mean.describe().to_string())


if __name__ == "__main__":
    main()
