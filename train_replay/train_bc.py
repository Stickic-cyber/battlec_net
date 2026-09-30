#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Behavioural-cloning (imitation learning) trainer for Battlecode 2026 "Snake".

Builds per-dragon observations from the replays (see build_features.py) for one
team and trains a small MLP to predict the expert's action:
    0=N  1=E  2=S  3=W  (move)
    4=SPLIT  5=SUICIDE
from the dragon's 7x7 wrapped vision window (7 channels) plus a couple of
scalars.

Model is tiny and CPU-friendly. Reports train/val accuracy and the per-class
action distribution (so you can see how imbalanced splitting/suicide are).
Saves:
  * <out>-obs.npy, <out>-scalar.npy, <out>-labels.npy  (the dataset)
  * <out>-model.pt  (state_dict + class distribution + config)

Usage (base Python has torch):
    python train_bc.py <folder/*.replay or replay> --team 0 --out team_A
    python train_bc.py <folder/*.replay or replay> --team 1 --out team_B
    # optionally rebuild the feature arrays first and just train:
    python train_bc.py --load team_A.npz --out team_A_model
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_features as bf

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

N_CLASSES = 6


class BCNet(nn.Module):
    """Small MLP over the flattened 7x7x7 observation + scalars."""

    def __init__(self, nch=bf.NCH, win=7, nscalar=2):
        super().__init__()
        self.win = win
        in_ch = nch * win * win + nscalar
        self.fc = nn.Sequential(
            nn.Linear(in_ch, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, obs, scalar):
        b = obs.shape[0]
        x = obs.reshape(b, -1)
        si = scalar.reshape(b, -1)
        return self.fc(torch.cat([x, si], dim=1))


def load_or_build(src, team, out_prefix):
    obs_path = out_prefix + "-obs.npy"
    scalar_path = out_prefix + "-scalar.npy"
    labels_path = out_prefix + "-labels.npy"
    if all(os.path.exists(p) for p in (obs_path, scalar_path, labels_path)):
        obs = np.load(obs_path)
        scalar = np.load(scalar_path)
        labels = np.load(labels_path)
        return obs, scalar, labels
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.replay")))
    else:
        files = [src]
    obs, scalar, labels, _ = bf.build(files, team)
    np.save(obs_path, obs)
    np.save(scalar_path, scalar)
    np.save(labels_path, labels)
    return obs, scalar, labels


def load_policy(model_path):
    """Load a saved policy; returns a callable obs(7,7,7)->action index
    and the label->name map."""
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = BCNet(nch=cfg["nch"], win=cfg["win"], nscalar=2)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    mu, sd = torch.tensor(cfg["mu"]), torch.tensor(cfg["sd"])
    names = ["N", "E", "S", "W", "SPLIT", "SUICIDE"]

    def predict(obs_patch, scalar):
        o = torch.tensor(obs_patch, dtype=torch.float32)[None]
        s = (torch.tensor(scalar, dtype=torch.float32)[None] - mu) / sd
        with torch.no_grad():
            p = net(o, s)[0].softmax(0)
        return int(p.argmax()), {names[i]: float(p[i]) for i in range(6)}

    return predict, names, ck["config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="replay file or folder; unused with --load")
    ap.add_argument("--team", type=int, help="0=A 1=B (not needed with --predict)")
    ap.add_argument("--out", default="bc_model")
    ap.add_argument("--load", help="load a saved dataset .npz (obs,scalar,labels) instead of building")
    ap.add_argument("--predict", metavar="MODEL.pt", help="load a trained model and print a few sample predictions")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not HAS_TORCH:
        sys.exit("torch not available in this interpreter; use the base Python that has torch")

    if args.predict:
        predict, names, cfg = load_policy(args.predict)
        nch, win = cfg["nch"], cfg["win"]
        rngm = np.random.default_rng(0)
        while True:
            o = rngm.random((nch, win, win)).astype(np.float32)
            s = np.array([0.5, 0.1], np.float32)
            a, probs = predict(o, s)
            print(f"action={names[a]} probs={ {k: round(v,3) for k,v in probs.items()} }")
            break
        print("load_policy ready. Use load_policy(model_path) in your code to act online.")
        return

    if args.load:
        d = np.load(args.load)
        obs, scalar, labels = d["obs"], d["scalar"], d["labels"]
    else:
        obs, scalar, labels = load_or_build(args.src, args.team, args.out)

    rng = np.random.default_rng(args.seed)
    n = len(labels)
    idx = rng.permutation(n)
    nval = max(1, int(0.2 * n))
    val_idx, tr_idx = idx[:nval], idx[nval:]
    print(f"team {args.team}: n={n} train={len(tr_idx)} val={len(val_idx)}")
    print("class distribution (train):",
          {i: int((labels[tr_idx] == i).sum()) for i in range(N_CLASSES)})

    # standardize scalars
    mu = scalar[tr_idx].mean(0, keepdims=True)
    sd = scalar[tr_idx].std(0, keepdims=True) + 1e-6
    scalar = (scalar - mu) / sd

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BCNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    Xo = torch.tensor(obs, dtype=torch.float32)
    Xs = torch.tensor(scalar, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.long)

    def eval_set(i):
        model.eval()
        with torch.no_grad():
            logits = model(Xo[i].to(device), Xs[i].to(device))
            pred = logits.argmax(1).cpu().numpy()
        return (pred == labels[i]).mean()

    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        tot, corr = 0, 0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            lo = model(Xo[b].to(device), Xs[b].to(device))
            loss = lossf(lo, Y[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            corr += int((lo.argmax(1).cpu().numpy() == labels[b]).sum()); tot += len(b)
        vacc = eval_set(val_idx)
        tacc = corr / tot
        print(f"epoch {ep+1:2d}: train_acc={tacc:.4f} val_acc={vacc:.4f}")

    # final metrics
    vacc = eval_set(val_idx); tacc = eval_set(tr_idx)
    model.eval()
    with torch.no_grad():
        pred = model(Xo.to(device), Xs.to(device)).argmax(1).cpu().numpy()
    from collections import Counter
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nconfusion matrix (all data):\n", confusion_matrix(labels, pred))
        print(classification_report(labels, pred, labels=list(range(N_CLASSES)),
                                    target_names=["N", "E", "S", "W", "SPLIT", "SUICIDE"],
                                    zero_division=0))
    except Exception:
        print("\n(sklearn unavailable; showing raw predictions vs labels)")

    torch.save({
        "state_dict": model.state_dict(),
        "class_counts": Counter(int(x) for x in labels),
        "config": {"team": args.team, "n_class": N_CLASSES, "nch": bf.NCH,
                   "win": 7, "mu": mu, "sd": sd, "n": n},
    }, args.out + "-model.pt")
    np.savez(args.out + "-dataset.npz", obs=obs, scalar=scalar, labels=labels)
    print(f"\nsaved model -> {args.out}-model.pt ; dataset -> {args.out}-dataset.npz")
    print(f"final val_acc={vacc:.4f} train_acc={tacc:.4f}  (majority-vote baseline = "
          f"{max(np.bincount(labels))/n:.3f})")


if __name__ == "__main__":
    main()
