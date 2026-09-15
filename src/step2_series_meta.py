"""Stage 2: read DICOM headers -> per-series geometry table.

Needed because the XML gives in-plane coordinates in PIXELS and z in MILLIMETRES.
Without PixelSpacing the two cannot be compared, which is the unit bug in the
original pipeline. Also records slice thickness and spacing so stage 4 can
resample to a fixed physical field of view.
"""
import argparse, io, os, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import pydicom


def read_series(zip_path):
    """Return (meta_dict, slice_rows) for one series zip."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except Exception as e:
        return {"zip_path": zip_path, "error": f"bad_zip:{e}"}, []

    slices = []
    with zf:
        for name in zf.namelist():
            if not name.lower().endswith(".dcm"):
                continue
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(name)), stop_before_pixels=True, force=True)
            except Exception:
                continue
            if getattr(ds, "Modality", None) != "CT":
                continue
            ipp = getattr(ds, "ImagePositionPatient", None)
            ps = getattr(ds, "PixelSpacing", None)
            if ipp is None or ps is None:
                continue
            slices.append({
                "member": name,
                "z_mm": float(ipp[2]),
                "origin_x_mm": float(ipp[0]),
                "origin_y_mm": float(ipp[1]),
                "row_spacing_mm": float(ps[0]),
                "col_spacing_mm": float(ps[1]),
                "rows": int(getattr(ds, "Rows", 0)),
                "cols": int(getattr(ds, "Columns", 0)),
                "thickness_mm": float(getattr(ds, "SliceThickness", np.nan) or np.nan),
                "slope": float(getattr(ds, "RescaleSlope", 1.0) or 1.0),
                "intercept": float(getattr(ds, "RescaleIntercept", 0.0) or 0.0),
                "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
                "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
                "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
                "patient_id": str(getattr(ds, "PatientID", "")),
            })

    if not slices:
        return {"zip_path": zip_path, "error": "no_ct_slices"}, []

    sdf = pd.DataFrame(slices).sort_values("z_mm").reset_index(drop=True)
    dz = np.diff(sdf.z_mm.values)
    meta = {
        "zip_path": zip_path,
        "series_uid": sdf.series_uid.mode().iat[0],
        "study_uid": sdf.study_uid.mode().iat[0],
        "patient_id": sdf.patient_id.mode().iat[0],
        "n_slices": len(sdf),
        "row_spacing_mm": float(sdf.row_spacing_mm.median()),
        "col_spacing_mm": float(sdf.col_spacing_mm.median()),
        "thickness_mm": float(np.nanmedian(sdf.thickness_mm.values)),
        "z_step_mm": float(np.median(np.abs(dz))) if len(dz) else np.nan,
        "z_step_irregular": bool(len(dz) and (np.abs(np.abs(dz) - np.median(np.abs(dz))) > 0.01).any()),
        "z_min_mm": float(sdf.z_mm.min()),
        "z_max_mm": float(sdf.z_mm.max()),
        "rows": int(sdf.rows.mode().iat[0]),
        "cols": int(sdf.cols.mode().iat[0]),
        "spacing_isotropic_inplane": bool(abs(sdf.row_spacing_mm.median() - sdf.col_spacing_mm.median()) < 1e-6),
        "error": "",
    }
    sdf["zip_path"] = zip_path
    return meta, sdf[["zip_path", "series_uid", "member", "z_mm", "origin_x_mm", "origin_y_mm",
                      "row_spacing_mm", "col_spacing_mm", "rows", "cols",
                      "slope", "intercept", "sop_uid"]].to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--out-series", required=True)
    ap.add_argument("--out-slices", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    zips = sorted(os.path.join(args.dicom_dir, f)
                  for f in os.listdir(args.dicom_dir) if f.endswith(".zip"))
    print(f"series zips: {len(zips)}", flush=True)

    metas, slice_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(read_series, p): p for p in zips}
        for i, fut in enumerate(as_completed(futs), 1):
            m, rows = fut.result()
            metas.append(m)
            slice_rows.extend(rows)
            if i % 100 == 0:
                print(f"  {i}/{len(zips)}", flush=True)

    mdf = pd.DataFrame(metas)
    mdf.to_csv(args.out_series, index=False)
    pd.DataFrame(slice_rows).to_csv(args.out_slices, index=False)

    ok = mdf[mdf.error == ""]
    print("\n--- stage 2 summary ---")
    print(f"  series parsed ok        {len(ok)} / {len(mdf)}")
    print(f"  unique patients         {ok.patient_id.nunique()}")
    print(f"  total CT slices         {int(ok.n_slices.sum())}")
    print(f"  in-plane spacing mm     min {ok.row_spacing_mm.min():.3f}  median {ok.row_spacing_mm.median():.3f}  max {ok.row_spacing_mm.max():.3f}")
    print(f"  slice thickness mm      min {ok.thickness_mm.min():.3f}  median {ok.thickness_mm.median():.3f}  max {ok.thickness_mm.max():.3f}")
    print(f"  z step mm               min {ok.z_step_mm.min():.3f}  median {ok.z_step_mm.median():.3f}  max {ok.z_step_mm.max():.3f}")
    print(f"  irregular z spacing     {int(ok.z_step_irregular.sum())} series")
    print(f"  anisotropic in-plane    {int((~ok.spacing_isotropic_inplane).sum())} series")
    if (mdf.error != "").any():
        print("\n  errors:")
        print(mdf[mdf.error != ""].error.value_counts().to_string())
    print(f"\n  12 px in-plane tolerance spans {12*ok.row_spacing_mm.min():.1f} mm to {12*ok.row_spacing_mm.max():.1f} mm  <-- the original unit bug")


if __name__ == "__main__":
    main()
