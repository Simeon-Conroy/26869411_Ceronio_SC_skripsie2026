from common import *
from pathlib import Path
import A_Complementary_gainB as Comp
import B_Madgwick_GainB as madg
import C_Mahony_GainB as Maho

RESULTS_FOLDER = Path(__file__).resolve().parents[1] / "filter_results"

FILTERS_TO_RUN = {
    "Complementary": False,
    "Madgwick": False,
    "Mahony": True,
}

NO_NOISE_MONTE_RUNS = 1
NOISE_MONTE_RUNS = 50
RUNS = [
    {
        "name": "Standstill",
        "dataset": r"C:\Users\ceron\Downloads\Year 4\Skripsie\Main\Glove\datasets\Analysis\Standstill.h5",
    },
    {
        "name": "Short_no_WP",
        "dataset": r"C:\Users\ceron\Downloads\Year 4\Skripsie\Main\Glove\datasets\Analysis\Short_no_WP.h5",
    },
    {
        "name": "Short_WP",
        "dataset": r"C:\Users\ceron\Downloads\Year 4\Skripsie\Main\Glove\datasets\Analysis\Short_WP.h5",
    },
    {
        "name": "Long_no_WP",
        "dataset": r"C:\Users\ceron\Downloads\Year 4\Skripsie\Main\Glove\datasets\Analysis\long_no_wp.h5",
    },
    {
        "name": "Long_WP",
        "dataset": r"C:\Users\ceron\Downloads\Year 4\Skripsie\Main\Glove\datasets\Analysis\Long_WP.h5",
    }
]

def print_banner(message, border="="):
    print(f"\n{border * 80}")
    print(message)
    print(f"{border * 80}\n")

def run_one_dataset(dataset_path, run_name, run_index, total_runs, output_dir, noise_enabled, monte_runs):
    print_banner(
        f"DATASET RUN {run_index}/{total_runs} STARTED | RUN: {run_name}\n"
        f"DATASET: {dataset_path}\nOUTPUT FOLDER: {output_dir}\n"
        f"NOISE MODEL: {'ENABLED' if noise_enabled else 'DISABLED'} | MONTE RUNS: {monte_runs}"
    )

    if FILTERS_TO_RUN["Complementary"]:
        Comp.main(dataset_path, run_name, output_dir, noise_enabled, monte_runs)
        print_banner(f"DONE: COMPLEMENTARY FILTER | RUN: {run_name}")
    else:
        print_banner(f"SKIPPED: COMPLEMENTARY FILTER | RUN: {run_name}")

    if FILTERS_TO_RUN["Madgwick"]:
        madg.main(dataset_path, run_name, output_dir, noise_enabled, monte_runs)
        print_banner(f"DONE: MADGWICK FILTER | RUN: {run_name}")
    else:
        print_banner(f"SKIPPED: MADGWICK FILTER | RUN: {run_name}")

    if FILTERS_TO_RUN["Mahony"]:
        Maho.main(dataset_path, run_name, output_dir, noise_enabled, monte_runs)
        print_banner(f"DONE: MAHONY FILTER | RUN: {run_name}")
    else:
        print_banner(f"SKIPPED: MAHONY FILTER | RUN: {run_name}")

    print_banner(
        f"DATASET RUN {run_index}/{total_runs} DONE | RUN: {run_name}",
        border="#",
    )


def main():
    total_runs = len(RUNS)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    print_banner(f"FILTER BATCH STARTED | DATASETS: {total_runs}\nOUTPUT FOLDER: {RESULTS_FOLDER}", border="#")

    noise_enabled = False
    print_banner("STARTING NO-NOISE PASS", border="#")
    for run_index, run in enumerate(RUNS, start=1):
        run_one_dataset(run["dataset"], run["name"], run_index, total_runs, RESULTS_FOLDER, noise_enabled, NO_NOISE_MONTE_RUNS)
    print_banner("NO-NOISE PASS DONE", border="#")

    
    noise_enabled = True
    print_banner("STARTING GAUSSIAN NOISE PASS", border="#")
    for run_index, run in enumerate(RUNS, start=1):
        run_one_dataset(run["dataset"], run["name"], run_index, total_runs, RESULTS_FOLDER, noise_enabled, NOISE_MONTE_RUNS)
    print_banner("GAUSSIAN NOISE PASS DONE", border="#")


if __name__ == "__main__":
    main()
