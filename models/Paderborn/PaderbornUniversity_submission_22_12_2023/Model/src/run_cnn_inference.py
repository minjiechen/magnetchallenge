"""Run convolutional neural networks inference on new 5 materials"""
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from joblib import delayed, Parallel
from pprint import pprint
import torch
import json
from torchinfo import summary as ti_summary
import argparse

from utils.experiments import (
    get_stratified_fold_indices,
    engineer_features,
    get_bh_integral_from_two_mats,
)
from utils.metrics import calculate_metrics
from utils.topology import TCNWithScalarsAsBias, LossPredictor
from utils.data import (
    TRIALS_CSV_PATH,
    EXP_CSV_PATH,
    TRIALS_CSV_COLS,
    EXP_CSV_COLS,
    MODEL_SINK,
    PRED_SINK,
    load_new_materials,
    ALL_B_COLS,
    DATA_SOURCE,
)

from run_cnn_training import construct_tensor_seq2seq, FREQ_SCALE


pd.set_option("display.max_columns", None)


def run_inference(model_uid, df=None):
    """Run inference with a certain model.
    All corresponding testing data will be loaded up and processed accordingly.

    Args
    ----
    model_uid: str
        Model identification
    df : pandas DataFrame (optional, default=None)
        Skip reading Testing data and use provided DataFrame instead (for the sake of transparency)

    Returns
    -------
    p_pred_ser: pandas Series
        The power loss prediction as pandas Series.
        Call p_pred_ser.to_numpy() to get the numpy representation
    """
    # filter model in meta info
    exp_tab = pd.read_csv(EXP_CSV_PATH, dtype=EXP_CSV_COLS)
    trials_tab = pd.read_csv(
        TRIALS_CSV_PATH,
        dtype=TRIALS_CSV_COLS,
        parse_dates=["start_date", "end_date"],
    ).query(f"model_uid == '{model_uid}'")
    # only one row should be returned
    meta_ser = trials_tab.merge(exp_tab, on="experiment_uid").iloc[0]
    targeted_material = meta_ser.loc["material"]

    # load up model
    mdl = None
    for mdl_path in MODEL_SINK.glob(
        f"*{meta_ser.loc['experiment_uid']}*{model_uid}*.pt"
    ):
        mdl = torch.jit.load(mdl_path)
    assert mdl is not None, f"Model file {model_uid} not found in {MODEL_SINK}"
    mdl.eval()

    # load up data
    if df is None:
        ds = load_new_materials(training=False, filter_materials=targeted_material)
    else:
        ds = df
    ds = engineer_features(ds, with_b_sat=False)

    # construct tensors
    x_cols = [
        c
        for c in ds
        if c not in ["ploss", "kfold", "material"]
        and not c.startswith(("B_t_", "H_t_"))
    ]
    B_COLS = ALL_B_COLS[:: meta_ser.loc["subsample_factor"]]
    with open(DATA_SOURCE.parent.parent / "b_max_dict.json", "r") as fh:
        b_limit = json.load(fh)[targeted_material]
    with open(DATA_SOURCE.parent.parent / "h_max_dict.json", "r") as fh:
        h_limit = json.load(fh)[targeted_material]

    b_limit_per_profile = (
        np.abs(ds.loc[:, B_COLS].to_numpy()).max(axis=1).reshape(-1, 1)
    )
    h_limit = h_limit * b_limit_per_profile / b_limit
    b_limit_test_fold = b_limit
    b_limit_test_fold_pp = b_limit_per_profile
    h_limit_test_fold = h_limit
    with torch.inference_mode():
        val_tensor_ts, val_tensor_scalar = construct_tensor_seq2seq(
            ds,
            x_cols,
            b_limit_test_fold,
            h_limit_test_fold,
            b_limit_pp=b_limit_test_fold_pp,
            training_data=False,
        )
        # does a p predictor exist?
        predicts_p_directly = meta_ser.loc["predicts_p_directly"]
        if predicts_p_directly:
            # prepare torch tensors for normalization scales

            b_limit_test_fold_torch = torch.as_tensor(
                b_limit_test_fold, dtype=torch.float32
            )
            h_limit_test_fold_torch = torch.as_tensor(
                h_limit_test_fold, dtype=torch.float32
            )
            freq_scale_torch = torch.as_tensor(
                FREQ_SCALE, dtype=torch.float32
            )

            val_pred_p, val_pred_h = mdl(
                val_tensor_ts.permute(1, 2, 0),
                val_tensor_scalar,
                b_limit_test_fold_torch,
                h_limit_test_fold_torch,
                freq_scale_torch,
            )
        else:
            val_pred_h = mdl(
                val_tensor_ts.permute(1, 2, 0),
                val_tensor_scalar,
            ).permute(2, 0, 1)
        h_pred_df = pd.DataFrame(
            val_pred_h.squeeze().cpu().numpy().T * h_limit_test_fold,
            columns=[f"H_t_{i}" for i in range(len(B_COLS))],
        )
        p_pred_ser = pd.Series(
            np.exp(val_pred_p.squeeze().cpu().numpy()).astype(np.float32),
            name="ploss",
        )
    h_pred_df.to_csv(
        PRED_SINK / f"CNN_H_preds_test_{targeted_material}_{model_uid}.csv",
        index=False,
        header=False,
    )
    p_pred_ser.to_csv(
        PRED_SINK / f"CNN_P_preds_test_{targeted_material}_{model_uid}.csv",
        index=False,
        header=False,
    )
    return p_pred_ser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN Inference")
    parser.add_argument("model_uid", help="The model uid from trials_meta.csv")
    args = parser.parse_args()
    run_inference(args.model_uid)
