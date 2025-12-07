import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", default="target")
    p.add_argument("--cat-cols", nargs="*", default=[])
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=777)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    assert args.target in df.columns, f"target column '{args.target}' not found"

    cat_cols = args.cat_cols
    num_cols = [c for c in df.columns if c not in cat_cols + [args.target]]

    # Build X/y
    X_num = df[num_cols].astype(float) if len(num_cols) else None
    X_cat = df[cat_cols].astype(str) if len(cat_cols) else None
    y = df[args.target].astype(int).to_numpy()

    # Split: train / val / test with stratification
    df_idx = np.arange(len(df))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        df_idx, y, test_size=args.val_size + args.test_size, stratify=y, random_state=args.random_state
    )
    rel_test = args.test_size / (args.val_size + args.test_size) if (args.val_size + args.test_size) > 0 else 0.0
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=args.random_state
    )

    def sel(X, idx):
        if X is None: return None
        return X.iloc[idx].to_numpy()

    parts = {
        "train": (idx_train, y_train),
        "val":   (idx_val,   y_val),
        "test":  (idx_test,  y_test),
    }

    # Save npy files
    for split, (idx, yy) in parts.items():
        if X_num is not None:
            np.save(out / f"X_num_{split}.npy", sel(X_num, idx))
        if X_cat is not None:
            np.save(out / f"X_cat_{split}.npy", sel(X_cat, idx))
        np.save(out / f"y_{split}.npy", yy)

    info = {
        "task_type": "binclass",
        "n_classes": 2,
        "train_size": int(len(idx_train)),
        "val_size": int(len(idx_val)),
        "test_size": int(len(idx_test)),
        "n_num_features": int(len(num_cols)),
        "n_cat_features": int(len(cat_cols)),
        "name": "heart",
        "id": "heart--id"
    }
    (out / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    print("Saved dataset to:", out.resolve())
    print("n_num_features:", info["n_num_features"], "n_cat_features:", info["n_cat_features"])
    print("sizes:", {k: len(v[0]) for k, v in parts.items()})

if __name__ == "__main__":
    main()
