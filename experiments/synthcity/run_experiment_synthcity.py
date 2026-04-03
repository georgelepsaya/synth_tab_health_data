import pandas as pd
import optuna
import sys
import torch
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins
from synthcity.metrics import Metrics
from synthcity.benchmark import Benchmarks
from synthcity.utils.optuna_sample import suggest_all

BATCH_SIZE = 8192
DEFAULT_N_TRIALS = 50
TUNE_GENERATE_COUNT = 5000  # cap synthetic rows during tuning; avoids slow generate+XGBoost on huge datasets


def log_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU detected, running on CPU")
    print()


def load_data(filepath, target_column, sep=","):
    X = pd.read_csv(filepath, sep=sep)
    if target_column not in X.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(X.columns)}"
        )
    loader = GenericDataLoader(X, target_column=target_column)
    return loader.train(), loader.test()


def run_experiment(model_name, filepath, target_col, out_path, n_trials, sep=","):
    log_device()
    train_loader, test_loader = load_data(filepath, target_col, sep=sep)
    print(f"Dataset: {filepath} ({len(train_loader) + len(test_loader)} rows)")
    print(f"Model: {model_name}, Trials: {n_trials}, Batch size: {BATCH_SIZE}\n")

    hp_space = Plugins().get(model_name).hyperparameter_space()

    def objective(trial: optuna.Trial):
        params = suggest_all(trial, hp_space)
        params["batch_size"] = BATCH_SIZE
        params["n_iter"] = 300

        try:
            model = Plugins().get(model_name, **params)
            model.fit(train_loader)
            synthetic = model.generate(count=min(len(train_loader), TUNE_GENERATE_COUNT))
            score_df = Metrics.evaluate(
                train_loader,
                synthetic,
                metrics={"performance": ["xgb"]},
                task_type="classification",
            )
            score = score_df["mean"].mean()
        except Exception as exc:
            print(f"  [trial {trial.number}] pruned: {exc}", file=sys.stderr)
            raise optuna.TrialPruned()

        print(f"  [trial {trial.number}] score={score:.6f}")
        return score

    print("=== Optimizing hyperparameters (TSTR with XGBoost) ===")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params["batch_size"] = BATCH_SIZE
    best_params.pop("n_iter", None)  # remove tuning shortcut; use model default for final eval

    print("\n=== Best parameters ===")
    print(best_params)

    print("\n=== Evaluating all metrics with best parameters ===")
    report = Benchmarks.evaluate(
        [("test", model_name, best_params)],
        train_loader,
        test_loader,
        repeats=1,
        task_type="classification",
    )
    Benchmarks.print(report)
    report["test"].to_csv(out_path)
    print(f"\nMetrics report saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: run_experiment_synthcity.py <model_name> <dataset_path> "
            "<output_path> [target_col] [n_trials]",
            file=sys.stderr,
        )
        sys.exit(1)

    model_name = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    target_col = sys.argv[4] if len(sys.argv) > 4 else "target"
    n_trials = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_N_TRIALS
    sep = sys.argv[6] if len(sys.argv) > 6 else ","

    run_experiment(model_name, dataset_path, target_col, output_path, n_trials, sep=sep)
