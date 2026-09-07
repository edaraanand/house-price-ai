from itertools import product

from train import main

param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [300, 600],
}

def generate_params():
    keys = param_grid.keys()

    for values in product(*param_grid.values()):
        yield dict(zip(keys, values))


def main_tuning():

    total_experiments = 18

    for i, params in enumerate(generate_params(), start=1):

        run_name = (
            f"tuning-{i:02d}"
            f"-depth={params['max_depth']}"
            f"-lr={params['learning_rate']}"
            f"-estimators={params['n_estimators']}"
        )

        print()
        print("=" * 60)
        print(f"EXPERIMENT {i}/{total_experiments}")
        print("=" * 60)
        print(f"Run name: {run_name}")
        print(f"Parameters: {params}")

        result = main(
            param_overrides=params,
            run_name=run_name,
            register_model=False,
            run_type="hyperparameter_tuning",
        )

        print(f"Run ID: {result['run_id']}")
        print(f"RMSE:   {result['metrics']['rmse']:.4f}")
        print(f"MAE:    {result['metrics']['mae']:.4f}")
        print(f"R2:     {result['metrics']['r2']:.4f}")


if __name__ == "__main__":
    main_tuning()
