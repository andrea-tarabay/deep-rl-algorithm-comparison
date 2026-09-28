import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDIES = {
    "expl_noise": [
        ("exploration noise = 0.05", "expl_noise_0.05"),
        ("exploration noise = 0.10 (default)", "default"),
        ("exploration noise = 0.20", "expl_noise_0.20"),
    ],
    "policy_noise": [
        ("target policy noise = 0.00", "policy_noise_0.00"),
        ("target policy noise = 0.10", "policy_noise_0.10"),
        ("target policy noise = 0.20 (default)", "default"),
    ],
    "policy_freq": [
        ("policy update frequency d = 1", "policy_freq_1"),
        ("policy update frequency d = 2 (default)", "default"),
        ("policy update frequency d = 4", "policy_freq_4"),
    ],

}


def read_experiment(env_name, folder):
    files = [
        f for f in os.listdir(folder)
        if f.startswith(env_name + "_seed") and f.endswith(".csv")
    ]

    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    all_data = []

    for f in files:
        path = os.path.join(folder, f)
        df = pd.read_csv(path)
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def summarize_curve(data, threshold=None):
    stats = data.groupby("timestep")["avg_return"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(0.0)

    final_row = stats.iloc[-1]
    best_row = stats.loc[stats["mean"].idxmax()]

    timesteps = stats["timestep"].to_numpy()
    means = stats["mean"].to_numpy()

    if timesteps[-1] > timesteps[0]:
        auc = np.trapezoid(means, timesteps) / (timesteps[-1] - timesteps[0])
    else:
        auc = means[-1]

    last_points = stats.tail(5)

    summary = {
        "final_mean": final_row["mean"],
        "final_std": final_row["std"],
        "best_mean": best_row["mean"],
        "best_timestep": best_row["timestep"],
        "auc_mean": auc,
        "last5_mean": last_points["mean"].mean(),
        "last5_std_over_time": last_points["mean"].std(),
    }

    if threshold is not None:
        reached = stats[stats["mean"] >= threshold]
        if len(reached) > 0:
            summary["steps_to_threshold"] = reached.iloc[0]["timestep"]
        else:
            summary["steps_to_threshold"] = np.nan

    return stats, summary


def plot_study(env_name, study_name, results_root, plots_root, tables_root, threshold):
    experiments = STUDIES[study_name]

    os.makedirs(os.path.join(plots_root, env_name), exist_ok=True)
    os.makedirs(os.path.join(tables_root, env_name), exist_ok=True)

    plt.figure(figsize=(10, 6))

    summary_rows = []

    for label, folder_name in experiments:
        folder = os.path.join(results_root, env_name, folder_name)

        if not os.path.exists(folder):
            print("Skipping missing folder:", folder)
            continue

        data = read_experiment(env_name, folder)
        stats, summary = summarize_curve(data, threshold=threshold)

        std = stats["std"]

        plt.plot(stats["timestep"], stats["mean"], label=label)
        plt.fill_between(
            stats["timestep"],
            stats["mean"] - std,
            stats["mean"] + std,
            alpha=0.15,
        )

        row = {"study": study_name, "setting": label, "folder": folder_name}
        row.update(summary)
        summary_rows.append(row)

    plt.xlabel("Environment steps")
    plt.ylabel("Average return")
    plt.title(f"{env_name} TD3 {study_name} study")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(plots_root, env_name, f"{env_name}_{study_name}.png")
    plt.savefig(plot_path, dpi=200)
    print("saved plot:", plot_path)

    summary_df = pd.DataFrame(summary_rows)
    table_path = os.path.join(tables_root, env_name, f"{env_name}_{study_name}_summary.csv")
    summary_df.to_csv(table_path, index=False)
    print("saved summary table:", table_path)

    print(summary_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument(
        "--study",
        type=str,
        required=True,
        choices=["expl_noise", "policy_noise", "policy_freq", "tau", "batch_size"],
    )
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--plots_root", type=str, default="plots")
    parser.add_argument("--tables_root", type=str, default="tables")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    plot_study(
        env_name=args.env,
        study_name=args.study,
        results_root=args.results_root,
        plots_root=args.plots_root,
        tables_root=args.tables_root,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()