"""Data and file handling.
Two CSV files will be used as database: experiments_meta.csv and trials_meta.csv.
These tables contain aggregated information about model training results and
training parameters involved for each model. Most importantly, they declare
unique IDs for each experiment and model, which helps to identify dumped
arrays of predictions and learning curves."""
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import delayed, Parallel
import csv

NEW_MATS_ROOT_PATH = Path.cwd().parent / "data" / "input" / "test"
ALL_B_COLS = [f"B_t_{k}" for k in range(1024)]
ALL_H_COLS = [f"H_t_{k}" for k in range(1024)]
EXP_CSV_PATH = Path.cwd().parent / "data" / "output" / "experiments_meta.csv"
TRIALS_CSV_PATH = EXP_CSV_PATH.parent / "trials_meta.csv"
TRIALS_CSV_COLS = {
    "experiment_uid": str,
    "material": str,
    "model_uid": str,
    "seed": int,
    "fold": int,
    "avg_rel_err": np.float32,
    "95-perc_rel_err": np.float32,
    "99-perc_rel_err": np.float32,
    "max_rel_err": np.float32,
    "start_date": str,
    "end_date": str,
}
EXP_CSV_COLS = {
    "experiment_uid": str,
    "debug": bool,
    "subsample_factor": int,
    "batch_size": int,
    "n_folds": int,
    "predicts_p_directly": bool,
    "n_epochs": int,
    "tag": str,
    'model_size': int,
}
DATA_SOURCE = Path.cwd().parent / "data" / "input" / "raw"
PROC_SOURCE = DATA_SOURCE.parent / "processed"
PRED_SINK = DATA_SOURCE.parent.parent / "output"
MODEL_SINK = PRED_SINK.parent / "models"


def load_material_csv_files_and_generate_pandas_df(mat_folder_path, training=True):
    mat_lbl = mat_folder_path.name.split(" ")[-1]
    b_df = pd.read_csv(mat_folder_path / "B_Field.csv", header=0, names=ALL_B_COLS)
    freq = pd.read_csv(mat_folder_path / "Frequency.csv", header=0)
    temp = pd.read_csv(mat_folder_path / "Temperature.csv", header=0)

    if training:
        h_df = pd.read_csv(mat_folder_path / "H_Field.csv", header=0, names=ALL_H_COLS)
        ploss = pd.read_csv(mat_folder_path / "Volumetric_Loss.csv", header=0)
        return pd.concat([b_df, h_df], axis=1).assign(
            freq=freq, temp=temp, ploss=ploss, material=mat_lbl
        )
    else:
        return b_df.assign(freq=freq, temp=temp, material=mat_lbl)


def load_new_materials(training=True, filter_materials=None):
    """Parallel reading of new materials CSV files.

    Parameters
    ----------
    training: Bool
        Load training (True) or test (False) data
    filter_materials: list or str, default: None (all materials)
        Materials to filter. Default is to load all materials.
        E.g.: ['A', 'D'] or ('B', 'C') or 'C'
    """
    folder_path = NEW_MATS_ROOT_PATH / ("Training" if training else "Testing")
    if filter_materials is None:
        with Parallel(n_jobs=5) as prll:
            mats_l = prll(
                delayed(load_material_csv_files_and_generate_pandas_df)(
                    mat_folder, training
                )
                for mat_folder in sorted(folder_path.glob("Material*"))
            )
    else:
        if isinstance(filter_materials, str):
            filter_materials = [filter_materials]
        assert isinstance(
            filter_materials, (list, tuple)
        ), "filter_materials must be list, tuple, str, or None"
        assert all(mat in "ABCDE" for mat in filter_materials), \
            f"load_new_materials() only loads materials ABCDE. Wrong requested dataset: {filter_materials}"
        mats_l = [
            load_material_csv_files_and_generate_pandas_df(mat_folder, training)
            for mat_folder in folder_path.glob("Material*")
            if mat_folder.name.endswith(tuple(filter_materials))
        ]

    return pd.concat(mats_l, axis=0, ignore_index=True)


def write_meta_info_to_csv(info_l, table="experiments"):
    """Boilerplate for writing pandas DataFrame meta information to
    CSV files, which are initialized with a header if they do not exist."""
    assert table in ["experiments", "trials"], f"table arg wrong: {table}"
    tab_path = EXP_CSV_PATH if table == "experiments" else TRIALS_CSV_PATH
    tab_cols = EXP_CSV_COLS if table == "experiments" else TRIALS_CSV_COLS
    tab_cols = list(tab_cols.keys())
    do_add_header = not tab_path.exists()
    with open(tab_path, "a") as fh:
        writer = csv.writer(fh)
        if do_add_header:
            writer.writerow(tab_cols)
        meta_df = pd.DataFrame(info_l)
        writer.writerows(meta_df.loc[:, tab_cols].to_numpy().tolist())


def bookkeeping(logs, debug=False, experiment_info=None):
    """Save relevant results to disk. This includes:
    * H (and P) predictions,
    * models as .pt files,
    * learning curves,
    * meta info as CSV.

    Parameters
    ----------
    logs: dict of dicts
        Return of main() from run_cnn.py
    debug: Bool
        Whether debug mode is active
    experiment_info: dict, default: None
        Arbitrary additional info involved in an experiment. Note that
        each key must exist in EXP_CSV_COLS.
    """
    # prepare folder structure for sinks
    PRED_SINK.mkdir(parents=True, exist_ok=True)
    MODEL_SINK.mkdir(parents=True, exist_ok=True)
    experiment_info = experiment_info or {}
    exp_uid = logs["experiment_uid"]
    print(f"Overall Score (Experiment {exp_uid})")
    mat_logs = logs["results_per_material"]
    performances_df = pd.DataFrame(
        {
            material: {  # TODO add quantiles of rel error, as is requested in report
                f"seed_{i}": mm["performance"]["percentile_95_rel_err"]
                for i, mm in enumerate(seed_logs_l)
            }
            for material, seed_logs_l in mat_logs.items()
        }
    )
    print(performances_df)  # (#seeds, #materials)
    best_seeds = np.argmin(performances_df.to_numpy(), axis=0)
    best_performances = {}
    print(f"Stats for best seed per material:")
    for best_seed_i, (mat_lbl, seed_logs_l) in zip(best_seeds, mat_logs.items()):
        best_seed_perf = seed_logs_l[best_seed_i]["performance"]
        print(
            f"Rel. Err. Distribution {mat_lbl} seed {best_seed_i}: Avg={100*best_seed_perf['avg-abs-rel-err']:.3f}%, "
            f"95-Prct={100*best_seed_perf['percentile_95_rel_err']:.1f}%, 99-Prct={100*best_seed_perf['percentile_99_rel_err']:.1f}%, "
            f"Max={100*best_seed_perf['l_infty']:.2f}%"
        )
        best_performances[mat_lbl] = {
            "seed": best_seed_i,
            "avg_rel_err": best_seed_perf["avg-abs-rel-err"],
            "95-perc_rel_err": best_seed_perf["percentile_95_rel_err"],
            "99-perc_rel_err": best_seed_perf["percentile_99_rel_err"],
            "max_rel_err": best_seed_perf["l_infty"],
        }

    # store predictions for post-processing
    print("Write predictions to disk..")
    pd.concat(
        [
            seed_logs_l[best_seed_i]["results_df"]
            .loc[
                :,
                [
                    c
                    for c in seed_logs_l[best_seed_i]["results_df"]
                    if c.startswith("h_pred_")
                ],
            ]
            .assign(material=material)
            for best_seed_i, (material, seed_logs_l) in zip(
                best_seeds, mat_logs.items()
            )
        ],
        ignore_index=True,
    ).to_csv(
        PRED_SINK / f"CNN_H_preds_{exp_uid}{'_debug' if debug else ''}.csv.zip",
        index=False,
    )
    pd.concat(
        [
            seed_logs_l[best_seed_i]["results_df"]
            .loc[:, ["pred"]]
            .assign(material=material)
            for best_seed_i, (material, seed_logs_l) in zip(
                best_seeds, mat_logs.items()
            )
        ],
        ignore_index=True,
    ).to_csv(
        PRED_SINK / f"CNN_P_preds_{exp_uid}{'_debug' if debug else ''}.csv.zip",
        index=False,
    )

    # store info to disk
    print("Write models as jit-script to disk..")
    seed_learning_trends_l = []
    trials_info_l = []
    end_date = pd.Timestamp.now().round(freq="S")
    for mat_lbl, seed_logs_l in mat_logs.items():
        for seed_i, seed_log in enumerate(seed_logs_l):
            n_folds = len(seed_log["model_scripted"])
            if 'model_size' not in experiment_info:
                experiment_info['model_size'] = seed_log['model_size']
            # construct pd DataFrame for learning trend of seed and material
            log_keys_to_store_l = ["loss_trends_train_h", "loss_trends_val_h"]
            has_predicted_P_directly = "loss_trends_train_p" in list(seed_log.keys())
            if has_predicted_P_directly:
                log_keys_to_store_l += ["loss_trends_train_p", "loss_trends_val_p"]
            seed_learning_trends_l.append(
                pd.concat(
                    [
                        pd.DataFrame(
                            seed_log[ks],
                            columns=[f"{ks}_fold_{i}" for i in range(n_folds)],
                        )
                        for ks in log_keys_to_store_l
                    ],
                    axis=1,
                ).assign(seed=seed_i, material=mat_lbl)
            )
            # store jitted models
            for fold_i, (scripted_mdl, mdl_uid) in enumerate(
                zip(seed_log["model_scripted"], seed_log["model_uids"])
            ):
                scripted_mdl.save(
                    MODEL_SINK
                    / (
                        f"cnn_{mat_lbl}_experiment_{exp_uid}_model_{mdl_uid}_"
                        f"seed_{seed_i}_fold_{fold_i}.pt"
                    )
                )
                # track meta info
                trials_info_l.append(
                    {
                        "experiment_uid": exp_uid,
                        "material": mat_lbl,
                        "model_uid": mdl_uid,
                        "seed": seed_i,
                        "fold": fold_i,
                        "avg_rel_err": seed_log["performance"]["avg-abs-rel-err"],
                        "95-perc_rel_err": seed_log["performance"][
                            "percentile_95_rel_err"
                        ],
                        "99-perc_rel_err": seed_log["performance"][
                            "percentile_99_rel_err"
                        ],
                        "max_rel_err": seed_log["performance"]["l_infty"],
                        "start_date": seed_log["start_time"],
                        "end_date": end_date,
                    }
                )

    print("Write learning trends to disk ..")
    pd.concat(seed_learning_trends_l, axis=0, ignore_index=True).to_csv(
        PRED_SINK / f"learning_curves_cnn_{exp_uid}{'_debug' if debug else ''}.csv.zip",
        index=False,
    )
    print("Write meta info to disk..")
    exp_info_l = [{**{"experiment_uid": exp_uid, "debug": debug}, **experiment_info}]
    write_meta_info_to_csv(exp_info_l, table="experiments")
    write_meta_info_to_csv(trials_info_l, table="trials")
    print("done.")
