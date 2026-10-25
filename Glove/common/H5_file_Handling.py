import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import h5py
import numpy as np

def choose_file():
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Select HDF5 Recording",
        filetypes=[("HDF5 Files", "*.h5")],
    )
    root.destroy()
    return filename

def Save_File_H5():

    root = tk.Tk()
    root.withdraw()

    filename = filedialog.asksaveasfilename(
        title="Save HDF5 Recording",
        filetypes=[("HDF5 Files", "*.h5")],
    )

    root.destroy()

    return filename

def result_filename(filter_name, run_name, output_dir="filter_results"):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_name)).strip("_")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{filter_name}_{safe_name}.h5" if safe_name else f"{filter_name}_Results.h5"
    return str(output_path / filename)

def save_results_h52(results_main, joint_names, filename):
    scalar_keys = [k for k in results_main[0].keys() if not k.endswith("_pj_mean")
                   and not k.endswith("_pj_std")]
    pj_keys = [k for k in results_main[0].keys() if k.endswith("_pj_mean")
               or k.endswith("_pj_std")]

    with h5py.File(filename, "w") as f:
        for k in scalar_keys:
            f.create_dataset(k, data=[r[k] for r in results_main])

        for k in pj_keys:
            stacked = np.stack([r[k] for r in results_main], axis=0)
            f.create_dataset(k, data=stacked)

        f.create_dataset(
            "joint_names",
            data=np.array(joint_names, dtype=h5py.string_dtype(encoding="utf-8")),
        )

    print(f"Saved {filename}")