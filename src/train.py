"""Train one (model, fold, seed) and write per-nodule held-out predictions.

Every configuration gets the same budget: 50 epochs, patience 8 on validation
mean MAE, AdamW with encoder lr 5e-5 and head lr 1e-4, batch 4. The test fold is
opened once, after the best validation checkpoint is restored.
"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

import models as M


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Nodules(Dataset):
    def __init__(self, df, train=False, hflip=0.5, unsorted=False):
        self.df = df.reset_index(drop=True)
        self.train, self.hflip, self.unsorted = train, hflip, unsorted

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        patch = np.load(r.patch_path).astype(np.float32)          # (S, H, W)
        if self.train and np.random.rand() < self.hflip:
            patch = patch[:, :, ::-1].copy()                      # left-right only
        cols = ("raw_score_1","raw_score_2","raw_score_3","raw_score_4") if self.unsorted \
               else ("score_1","score_2","score_3","score_4")
        scores = np.array([r[c] for c in cols], dtype=np.float32)
        ordinal = (scores[:, None] > np.arange(1, M.N_SCORES)[None, :]).astype(np.float32)
        dist = np.zeros(M.N_SCORES, dtype=np.float32)
        for s in scores:
            dist[int(s) - 1] += 1.0
        dist /= dist.sum()
        return {
            "x": torch.from_numpy(patch).unsqueeze(1),            # (S, 1, H, W)
            "scores": torch.from_numpy(scores),
            "ordinal": torch.from_numpy(ordinal),
            "dist": torch.from_numpy(dist),
            "score_mean": torch.tensor(float(r.score_mean)),
            "score_var_norm": torch.tensor(float(r.score_var_norm)),
            "cluster_id": r.cluster_id, "patient_id": r.patient_id,
        }


def compute_loss(out, batch, cfg, dev):
    lam_s = cfg.get("lambda_score", 0.0)
    tgt_scores = batch["scores"].to(dev)
    tgt_mean = batch["score_mean"].to(dev)
    tgt_dist = batch["dist"].to(dev)
    tgt_var = batch["score_var_norm"].to(dev)
    parts, total = {}, 0.0
    kind = cfg["head"]

    if kind == "ordinal_multi":
        l_ord = F.binary_cross_entropy_with_logits(out["ordinal_logits"], batch["ordinal"].to(dev))
        total = total + l_ord; parts["ord"] = float(l_ord)
        pred_dist = torch.softmax(-((out["scores"].unsqueeze(-1)
                    - torch.arange(1, M.N_SCORES + 1, device=dev).view(1, 1, -1)) ** 2) / (2 * 0.5 ** 2), dim=-1).mean(1)
        l_dist = F.kl_div(torch.log(pred_dist.clamp_min(1e-8)), tgt_dist, reduction="batchmean")
        total = total + 0.2 * l_dist; parts["dist"] = float(l_dist)
        l_unc = F.mse_loss(out["pred_var_norm"], tgt_var)
        total = total + 0.2 * l_unc; parts["unc"] = float(l_unc)
        if lam_s > 0:
            l_sc = F.smooth_l1_loss(out["scores"], tgt_scores)
            total = total + lam_s * l_sc; parts["score"] = float(l_sc)
    elif kind == "regress_multi":
        l_sc = F.smooth_l1_loss(out["scores"], tgt_scores)
        l_unc = F.mse_loss(out["pred_var_norm"], tgt_var)
        total = l_sc + 0.2 * l_unc
        parts.update(score=float(l_sc), unc=float(l_unc))
    elif kind == "ldl":
        l_dist = F.kl_div(torch.log(out["dist"].clamp_min(1e-8)), tgt_dist, reduction="batchmean")
        l_unc = F.mse_loss(out["pred_var_norm"], tgt_var)
        total = l_dist + 0.2 * l_unc
        parts.update(dist=float(l_dist), unc=float(l_unc))
    elif kind == "ordinal_single":
        rounded = torch.clamp(torch.round(tgt_mean), 1, M.N_SCORES)
        ordinal = (rounded.unsqueeze(1) > torch.arange(1, M.N_SCORES, device=dev).view(1, -1)).float()
        total = F.binary_cross_entropy_with_logits(out["ordinal_logits"].squeeze(1), ordinal)
        parts["ord"] = float(total)
    else:                                                            # scalar
        total = F.smooth_l1_loss(out["pred_mean"], tgt_mean)
        parts["score"] = float(total)
    return total, parts


@torch.no_grad()
def evaluate(model, loader, cfg, dev, collect=False):
    model.eval()
    rows, losses = [], []
    for b in loader:
        out = model(b["x"].to(dev))
        loss, _ = compute_loss(out, b, cfg, dev)
        losses.append(float(loss))
        pm = out["pred_mean"].cpu().numpy()
        pv = out["pred_var_norm"].cpu().numpy()
        sc = out["scores"].cpu().numpy()
        for i in range(len(pm)):
            r = {"cluster_id": b["cluster_id"][i], "patient_id": b["patient_id"][i],
                 "true_mean": float(b["score_mean"][i]), "true_var_norm": float(b["score_var_norm"][i]),
                 "pred_mean": float(pm[i]), "pred_var_norm": float(pv[i])}
            if collect:
                for k in range(sc.shape[1]):
                    r[f"pred_score_{k+1}"] = float(sc[i, k])
                for k in range(4):
                    r[f"true_score_{k+1}"] = float(b["scores"][i, k])
            rows.append(r)
    d = pd.DataFrame(rows)
    err = (d.pred_mean - d.true_mean).abs()
    m = {"loss": float(np.mean(losses)), "mean_mae": float(err.mean()),
         "mean_spearman": float(spearmanr(d.true_mean, d.pred_mean).statistic),
         "var_mae": float((d.pred_var_norm - d.true_var_norm).abs().mean()),
         "var_spearman": float(spearmanr(d.true_var_norm, d.pred_var_norm).statistic)}
    return m, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--model", required=True, choices=list(M.CONFIGS))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-slices", type=int, default=7)
    args = ap.parse_args()

    tag = f"{args.model}_f{args.fold}_s{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    done = os.path.join(args.out_dir, f"{tag}_metrics.json")
    if os.path.exists(done):
        print(f"{tag} already done, skipping"); return

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1000 * args.seed + args.fold)

    df = pd.read_csv(args.table)
    vcol = f"is_val_fold{args.fold}"
    te = df[df.fold == args.fold]
    tr = df[(df.fold != args.fold) & (~df[vcol])]
    va = df[(df.fold != args.fold) & (df[vcol])]
    assert not (set(tr.patient_id) & set(va.patient_id)) and not (set(tr.patient_id) & set(te.patient_id))

    cfg = M.CONFIGS[args.model]
    uns = bool(cfg.get("unsorted", False))
    mk = lambda d, t: DataLoader(Nodules(d, train=t, unsorted=uns), batch_size=args.batch_size,
                                 shuffle=t, num_workers=args.workers, drop_last=False)
    tl, vl, sl = mk(tr, True), mk(va, False), mk(te, False)

    model = M.build(args.model, n_slices=args.n_slices).to(dev)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc, "lr": 5e-5}, {"params": rest, "lr": 1e-4}], weight_decay=1e-4)

    best, best_ep, bad, ckpt = float("inf"), -1, 0, os.path.join(args.out_dir, f"{tag}.pt")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            opt.zero_grad()
            loss, _ = compute_loss(model(b["x"].to(dev)), b, cfg, dev)
            loss.backward(); opt.step()
        vm, _ = evaluate(model, vl, cfg, dev)
        print(f"  ep {ep+1:02d} val_mae {vm['mean_mae']:.4f} val_sp {vm['mean_spearman']:.4f}", flush=True)
        if vm["mean_mae"] < best:
            best, best_ep, bad = vm["mean_mae"], ep + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {ep+1}"); break

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    tm, preds = evaluate(model, sl, cfg, dev, collect=True)
    preds.insert(2, "seed", args.seed); preds.insert(2, "fold", args.fold)
    preds.insert(2, "model", args.model)
    preds.to_csv(os.path.join(args.out_dir, f"{tag}_preds.csv"), index=False)

    rec = {"model": args.model, "fold": args.fold, "seed": args.seed,
           "best_epoch": best_ep, "best_val_mae": best, "diverged": bool(best_ep <= 3),
           "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "minutes": round((time.time() - t0) / 60, 2), **{f"test_{k}": v for k, v in tm.items()}}
    with open(done, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
