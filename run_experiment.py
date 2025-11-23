import pandas as pd
import optuna
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


def run_experiment(model_name, filepath):
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

