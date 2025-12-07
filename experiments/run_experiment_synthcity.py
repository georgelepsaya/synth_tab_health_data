import os
import multiprocessing

num_cores = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

import torch
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)

print(f"Using {num_cores} CPU cores")

import pandas as pd
import optuna
import sys
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins
from synthcity.benchmark import Benchmarks
from synthcity.utils.optuna_sample import suggest_all


def load_data(filepath):
    X = pd.read_csv(filepath)
    loader = GenericDataLoader(
        X,
        target_column="target",
    )
    return loader.train(), loader.test()


def run_experiment(model_name, filepath, out_path):
    """Run optimization and evaluation of a model"""

    train_loader, test_loader = load_data(filepath)

    def objective(trial: optuna.Trial):
        hp_space = Plugins().get(model_name).hyperparameter_space()
        params = suggest_all(trial, hp_space)
        ID = f"trial_{trial.number}"
        try:
            report = Benchmarks.evaluate(
                [(ID, model_name, params)],
                train_loader,
                repeats=1,
                metrics={"stats": ["jensenshannon_dist"]},
            )
        except Exception:
            raise optuna.TrialPruned()

        score = report[ID]["mean"].mean()
        return score

    print("=== Optimizing hyperparameters ===")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    best_params = study.best_params

    print("\n=== Best parameters ===")
    print(best_params)

    print("\n=== Evaluating all metrics with best parameters ===")
    report = Benchmarks.evaluate(
        [("test", model_name, best_params)],
        train_loader,
        test_loader,
        repeats=1,
    )
    Benchmarks.print(report)
    report["test"].to_csv(out_path, index=False)
    print(f"Metrics report saved to {out_path}")


if __name__ == "__main__":
    model_name = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    run_experiment(model_name, dataset_path, output_path)
