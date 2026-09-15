"""Encoders, aggregators and heads for the MROU-Net study.

Every configuration shares the same 7-slice input, the same per-slice encoder
application and the same training budget. Only the named component changes, so
Table 1 and the ablation table are paired comparisons of one thing at a time.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

N_SCORES = 5          # malignancy ratings are 1..5
N_THRESH = N_SCORES - 1


# --------------------------------------------------------------------------- encoders

def _gray_stem(conv):
    """Replace a 3-channel conv with a 1-channel one, averaging the RGB weights."""
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=conv.bias is not None)
    new.weight.data[:, 0] = conv.weight.data.mean(dim=1)
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    return new


class Encoder(nn.Module):
    """2D backbone adapted to single-channel CT, returning a pooled feature vector."""

    def __init__(self, name="convnext_tiny", pretrained=True):
        super().__init__()
        self.name = name
        if name == "convnext_tiny":
            b = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            b.features[0][0] = _gray_stem(b.features[0][0])
            self.body, self.out_dim = b.features, 768
        elif name == "mobilenet_v3_large":
            b = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            b.features[0][0] = _gray_stem(b.features[0][0])
            self.body, self.out_dim = b.features, 960
        elif name == "efficientnet_b0":
            b = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            b.features[0][0] = _gray_stem(b.features[0][0])
            self.body, self.out_dim = b.features, 1280
        elif name == "vgg16":
            b = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT if pretrained else None)
            b.features[0] = _gray_stem(b.features[0])
            self.body, self.out_dim = b.features, 512
        elif name == "xception":
            import timm
            b = timm.create_model("legacy_xception", pretrained=pretrained, in_chans=1, num_classes=0)
            self.body, self.out_dim = b, b.num_features
            self.timm = True
        else:
            raise ValueError(f"unknown encoder {name}")
        self.timm = getattr(self, "timm", False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):                      # (B*S, 1, H, W)
        if self.timm:
            return self.body(x)
        return self.pool(self.body(x)).flatten(1)


# --------------------------------------------------------------------------- aggregation

class CenterGuidedAttention(nn.Module):
    """Centre slice is the query; all slices are keys and values."""

    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, feats):                  # (B, S, D)
        centre = feats[:, feats.shape[1] // 2]
        Q = self.q(centre).unsqueeze(1)
        A = torch.softmax((Q @ self.k(feats).transpose(1, 2)) * self.scale, dim=-1)
        return (A @ self.v(feats)).squeeze(1), A.squeeze(1)


class MeanPool(nn.Module):
    def forward(self, feats):
        return feats.mean(dim=1), None


# --------------------------------------------------------------------------- heads

# NOTE: bounded regression uses 1 + 4*sigmoid rather than clamp(1, 5). A hard clamp
# zeroes the gradient whenever the raw output falls outside the range, and since a
# freshly initialised linear layer outputs near 0, every such head died at step one
# and predicted the constant 1.0 forever.

def ordinal_to_score(logits):
    """logits (B, H, N_THRESH) -> expected score in [1, 5] and the threshold probs."""
    p = torch.sigmoid(logits)
    return 1.0 + p.sum(dim=-1), p


class Net(nn.Module):
    def __init__(self, encoder="convnext_tiny", pretrained=True, n_slices=7,
                 aggregation="attention", positional=True, head="ordinal_multi",
                 n_heads=4, dropout=0.2):
        super().__init__()
        self.cfg = dict(encoder=encoder, aggregation=aggregation, positional=positional,
                        head=head, n_heads=n_heads)
        self.encoder = Encoder(encoder, pretrained)
        d = self.encoder.out_dim
        self.positional = positional
        if positional:
            self.pos = nn.Parameter(torch.zeros(1, n_slices, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
        self.agg = CenterGuidedAttention(d) if aggregation == "attention" else MeanPool()
        self.proj = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(dropout))
        self.head_kind, self.n_heads = head, n_heads
        if head == "ordinal_multi":
            self.heads = nn.ModuleList([nn.Linear(512, N_THRESH) for _ in range(n_heads)])
        elif head == "regress_multi":
            self.heads = nn.ModuleList([nn.Linear(512, 1) for _ in range(n_heads)])
        elif head == "ordinal_single":
            self.heads = nn.ModuleList([nn.Linear(512, N_THRESH)])
        elif head == "scalar":
            self.heads = nn.ModuleList([nn.Linear(512, 1)])
        elif head == "ldl":
            self.heads = nn.ModuleList([nn.Linear(512, N_SCORES)])
        else:
            raise ValueError(head)

    def forward(self, x):                      # (B, S, 1, H, W)
        B, S = x.shape[:2]
        f = self.encoder(x.reshape(B * S, *x.shape[2:])).reshape(B, S, -1)
        if self.positional:
            f = f + self.pos[:, :S]
        z, attn = self.agg(f)
        z = self.proj(z)
        out = {"attn": attn}

        if self.head_kind in ("ordinal_multi", "ordinal_single"):
            logits = torch.stack([h(z) for h in self.heads], dim=1)
            scores, probs = ordinal_to_score(logits)
            out["ordinal_logits"], out["scores"] = logits, scores
        elif self.head_kind == "regress_multi":
            raw = torch.cat([h(z) for h in self.heads], dim=1)
            out["scores"] = 1.0 + 4.0 * torch.sigmoid(raw)
        elif self.head_kind == "scalar":
            out["scores"] = 1.0 + 4.0 * torch.sigmoid(self.heads[0](z))
        elif self.head_kind == "ldl":
            dist = torch.softmax(self.heads[0](z), dim=-1)
            centres = torch.arange(1, N_SCORES + 1, device=dist.device, dtype=dist.dtype)
            mean = (dist * centres).sum(-1)
            var = (dist * (centres.view(1, -1) - mean.view(-1, 1)) ** 2).sum(-1)
            out["dist"], out["scores"] = dist, mean.unsqueeze(1)
            out["pred_var_norm"] = var / 4.0

        s = out["scores"]
        out["pred_mean"] = s.mean(dim=1)
        if "pred_var_norm" not in out:
            out["pred_var_norm"] = (s.var(dim=1, unbiased=False) / 4.0 if s.shape[1] > 1
                                    else torch.zeros_like(out["pred_mean"]))
        return out


# --------------------------------------------------------------------------- registry

CONFIGS = {
    # proposed model and its ablations
    "mrou_full":        dict(encoder="convnext_tiny", aggregation="attention", positional=True,  head="ordinal_multi", lambda_score=0.05),
    "mrou_no_attn":     dict(encoder="convnext_tiny", aggregation="mean",      positional=True,  head="ordinal_multi", lambda_score=0.05),
    "mrou_no_pos":      dict(encoder="convnext_tiny", aggregation="attention", positional=False, head="ordinal_multi", lambda_score=0.05),
    "mrou_no_ordinal":  dict(encoder="convnext_tiny", aggregation="attention", positional=True,  head="regress_multi", lambda_score=0.05),
    "mrou_no_lscore":   dict(encoder="convnext_tiny", aggregation="attention", positional=True,  head="ordinal_multi", lambda_score=0.0),
    # the control that isolates aggregation: same supervision, mean pooling, no positional
    "mrou_meanpool_nopos": dict(encoder="convnext_tiny", aggregation="mean",   positional=False, head="ordinal_multi", lambda_score=0.05),
    # ablations of the FINAL model, which drops L_score. The variants above keep
    # L_score, so comparing them against mrou_no_lscore would change two things at once.
    "nols_no_attn":     dict(encoder="convnext_tiny", aggregation="mean",      positional=True,  head="ordinal_multi", lambda_score=0.0),
    "nols_no_pos":      dict(encoder="convnext_tiny", aggregation="attention", positional=False, head="ordinal_multi", lambda_score=0.0),
    "nols_no_ordinal":  dict(encoder="convnext_tiny", aggregation="attention", positional=True,  head="regress_multi", lambda_score=0.0),
    # Reviewer 1 asked explicitly for multi-output regression WITHOUT score sorting.
    # Same encoder, same aggregation, same heads; only the target order changes.
    "unsorted_regress": dict(encoder="convnext_tiny", aggregation="attention", positional=True, head="regress_multi", lambda_score=0.0, unsorted=True),
    # Head-matched baselines on the SAME aggregation as the proposed model, so the
    # comparison isolates the prediction head rather than confounding it with pooling.
    "ldl_attn":            dict(encoder="convnext_tiny", aggregation="attention", positional=True, head="ldl",            lambda_score=0.0),
    "single_ordinal_attn": dict(encoder="convnext_tiny", aggregation="attention", positional=True, head="ordinal_single", lambda_score=0.0),
    "scalar_attn":         dict(encoder="convnext_tiny", aggregation="attention", positional=True, head="scalar",         lambda_score=0.0),
    # task-matched baselines
    "ldl":              dict(encoder="convnext_tiny", aggregation="mean", positional=False, head="ldl",            lambda_score=0.0),
    "single_ordinal":   dict(encoder="convnext_tiny", aggregation="mean", positional=False, head="ordinal_single", lambda_score=0.0),
    # backbone baselines, single scalar target
    "convnext_scalar":  dict(encoder="convnext_tiny",       aggregation="mean", positional=False, head="scalar", lambda_score=0.0),
    "mobilenetv3":      dict(encoder="mobilenet_v3_large",  aggregation="mean", positional=False, head="scalar", lambda_score=0.0),
    "efficientnet_b0":  dict(encoder="efficientnet_b0",     aggregation="mean", positional=False, head="scalar", lambda_score=0.0),
    "vgg16":            dict(encoder="vgg16",               aggregation="mean", positional=False, head="scalar", lambda_score=0.0),
    "xception":         dict(encoder="xception",            aggregation="mean", positional=False, head="scalar", lambda_score=0.0),
}


def build(name, pretrained=True, n_slices=7):
    cfg = dict(CONFIGS[name])
    cfg.pop("lambda_score", None)
    cfg.pop("unsorted", None)
    return Net(pretrained=pretrained, n_slices=n_slices, **cfg)
