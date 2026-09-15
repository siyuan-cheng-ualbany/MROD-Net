"""Stage 1: LIDC XML -> reader-level nodule table.

One row per (series, reading session, nodule) with the malignancy rating and the
bounding geometry of the annotated contours, in pixel/DICOM coordinates.
Stage 3 converts those to millimetres using the series pixel spacing.
"""
import argparse, os, re, sys, zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

NS = {"n": "http://www.nih.gov"}


def _text(el, tag):
    if el is None:
        return None
    found = el.find(f"n:{tag}", NS)
    if found is None:
        found = el.find(tag)
    return found.text.strip() if found is not None and found.text else None


def _findall(el, tag):
    out = el.findall(f"n:{tag}", NS)
    return out if out else el.findall(tag)


def parse_one(xml_bytes, source_name):
    """Return (rows, series_uid, stats) for a single LIDC XML file."""
    rows, stats = [], {}

    def bump(k):
        stats[k] = stats.get(k, 0) + 1

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        bump("xml_parse_error")
        return rows, None, stats

    hdr = root.find("n:ResponseHeader", NS) or root.find("ResponseHeader")
    series_uid = _text(hdr, "SeriesInstanceUid")
    study_uid = _text(hdr, "StudyInstanceUID") or _text(hdr, "StudyInstanceUid")
    if series_uid is None:
        bump("no_series_uid")
        return rows, None, stats

    sessions = _findall(root, "readingSession")
    for reader_idx, sess in enumerate(sessions):
        for nod in _findall(sess, "unblindedReadNodule"):
            bump("reader_nodules_seen")
            nodule_id = _text(nod, "noduleID")
            ch = nod.find("n:characteristics", NS) or nod.find("characteristics")
            mal = _text(ch, "malignancy")
            if mal is None:
                bump("excluded_missing_malignancy")
                continue
            try:
                mal = int(float(mal))
            except (TypeError, ValueError):
                bump("excluded_invalid_malignancy")
                continue
            if not 1 <= mal <= 5:
                bump("excluded_out_of_range_malignancy")
                continue

            xs, ys, zs, n_roi = [], [], [], 0
            for roi in _findall(nod, "roi"):
                z = _text(roi, "imageZposition")
                try:
                    z = float(z)
                except (TypeError, ValueError):
                    continue
                pts = []
                for em in _findall(roi, "edgeMap"):
                    xc, yc = _text(em, "xCoord"), _text(em, "yCoord")
                    try:
                        pts.append((float(xc), float(yc)))
                    except (TypeError, ValueError):
                        continue
                if not pts:
                    continue
                n_roi += 1
                for xc, yc in pts:
                    xs.append(xc); ys.append(yc); zs.append(z)

            if not xs:
                bump("excluded_no_valid_spatial_points")
                continue

            xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
            bump("reader_nodules_retained")
            rows.append({
                "source_xml": source_name,
                "series_uid": series_uid,
                "study_uid": study_uid,
                "reader_idx": reader_idx,
                "n_readers_in_file": len(sessions),
                "nodule_id": nodule_id,
                "malignancy": mal,
                "n_roi": n_roi,
                "n_points": len(xs),
                # pixel coordinates
                "center_x_px": float(xs.mean()),
                "center_y_px": float(ys.mean()),
                "min_x_px": float(xs.min()), "max_x_px": float(xs.max()),
                "min_y_px": float(ys.min()), "max_y_px": float(ys.max()),
                "diam_x_px": float(xs.max() - xs.min()),
                "diam_y_px": float(ys.max() - ys.min()),
                # z is already in millimetres (ImagePositionPatient[2])
                "center_z_mm": float(zs.mean()),
                "min_z_mm": float(zs.min()), "max_z_mm": float(zs.max()),
                "z_span_mm": float(zs.max() - zs.min()),
            })
    return rows, series_uid, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", required=True)
    args = ap.parse_args()

    all_rows, total = [], {}
    n_files = 0
    with zipfile.ZipFile(args.xml_zip) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".xml")]
        print(f"xml files in archive: {len(names)}", flush=True)
        for name in names:
            n_files += 1
            rows, _, stats = parse_one(z.read(name), os.path.basename(name))
            all_rows.extend(rows)
            for k, v in stats.items():
                total[k] = total.get(k, 0) + v

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)

    total["xml_files_processed"] = n_files
    total["series_with_annotations"] = int(df.series_uid.nunique()) if len(df) else 0
    total["reader_level_rows_written"] = len(df)
    pd.DataFrame(sorted(total.items()), columns=["category", "count"]).to_csv(args.stats_out, index=False)

    print("\n--- stage 1 summary ---")
    for k, v in sorted(total.items()):
        print(f"  {k:38s} {v}")
    if len(df):
        print("\nreadings per series:")
        print(df.groupby("series_uid").reader_idx.nunique().value_counts().sort_index().to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
