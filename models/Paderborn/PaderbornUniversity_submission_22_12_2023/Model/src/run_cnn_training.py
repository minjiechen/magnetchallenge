"""Run convolutional neural networks training """
import pandas as pd
import numpy as np
from tqdm import trange
import torch
from torchinfo import summary as ti_summary
import random
from uuid import uuid4
import argparse

from utils.experiments import (
    get_stratified_fold_indices,
    get_bh_integral_from_two_mats,
    engineer_features,
)
from utils.metrics import calculate_metrics
from utils.topology import TCNWithScalarsAsBias, LossPredictor
from utils.data import (
    load_new_materials,
    ALL_B_COLS,
    ALL_H_COLS,
    bookkeeping,
    PROC_SOURCE,
)


pd.set_option("display.max_columns", None)

DEBUG = False
N_SEEDS = 5  # how often should the experiment be repeated with different random init
N_JOBS = 1  # how many processes should be working
N_EPOCHS = 2 if DEBUG else 5000  # how often should the full data set be iterated over
half_lr_at = [int(N_EPOCHS * 0.8)]  # halve learning rate after these many epochs
SUBSAMPLE_FACTOR = (
    8 if DEBUG else 1
)  # every n-th sample along the time axis is considered
FREQ_SCALE = 150_000  # in Hz
K_KFOLD = 1 if DEBUG else 4  # how many folds in cross validation
BATCH_SIZE = (
    4 if DEBUG else 64
)  # how many periods/profiles/measurements should be averaged across for a weight update
DO_PREDICT_P_DIRECTLY = True  # Whether to extend the topology to predict p loss with a parameterized model on top
B_COLS = ALL_B_COLS[::SUBSAMPLE_FACTOR]
H_COLS = ALL_H_COLS[::SUBSAMPLE_FACTOR]
H_PRED_COLS = [f"h_pred_{i}" for i in range(1024 // SUBSAMPLE_FACTOR)]
DEBUG_MATERIALS = {"old": ["3C90", "78"], "new": ["A", "B", "C", "D", "E"]}
TRAIN_ON_NEW_MATERIALS = True


def construct_tensor_seq2seq(
    df,
    x_cols,
    b_limit,
    h_limit,
    b_limit_pp=None,
    ln_ploss_mean=0,
    ln_ploss_std=1,
    training_data=True,
):
    """generate tensors with following shapes:
    For time series tensors (#time steps, #profiles/periods, #features),
    for scalar tensors (#profiles, #features)"""
    full_b = df.loc[:, B_COLS].to_numpy()
    if training_data:
        full_h = df.loc[:, H_COLS].to_numpy()
    df = df.drop(columns=[c for c in df if c.startswith(("H_t_", "B_t_", "material"))])
    assert len(df) > 0, "empty dataframe error"
    # put freq on first place since Architecture expects it there
    x_cols.insert(0, x_cols.pop(x_cols.index("freq")))
    X = df.loc[:, x_cols]
    # normalization
    full_b /= b_limit
    if training_data:
        full_h /= h_limit
    orig_freq = X.loc[:, ["freq"]].copy().to_numpy()
    X.loc[:, ["temp", "freq"]] /= np.array([75.0, FREQ_SCALE])
    X.loc[:, "freq"] = np.log(X.freq)
    other_cols = [c for c in x_cols if c not in ["temp", "freq"]]
    X.loc[:, other_cols] /= X.loc[:, other_cols].abs().max(axis=0)
    if training_data:
        # add p loss as target (only used as target when predicting p loss directly), must be last column
        X = X.assign(ln_ploss=(np.log(df.ploss) - ln_ploss_mean) / ln_ploss_std)
    # tensor list
    tens_l = []
    if b_limit_pp is not None:
        # add another B curve with different normalization
        per_profile_scaled_b = full_b * b_limit / b_limit_pp
        # add timeseries derivatives
        b_deriv = np.empty((full_b.shape[0], full_b.shape[1] + 2))
        b_deriv[:, 1:-1] = per_profile_scaled_b
        b_deriv[:, 0] = per_profile_scaled_b[:, -1]
        b_deriv[:, -1] = per_profile_scaled_b[:, 0]
        b_deriv = np.gradient(b_deriv, axis=1) * orig_freq
        b_deriv_sq = np.gradient(b_deriv, axis=1) * orig_freq
        b_deriv = b_deriv[:, 1:-1]
        b_deriv_sq = b_deriv_sq[:, 1:-1]
        tantan_b = -np.tan(0.9 * np.tan(per_profile_scaled_b)) / 6  # tan-tan feature
        tens_l += [
            torch.tensor(per_profile_scaled_b.T[..., np.newaxis], dtype=torch.float32),
            torch.tensor(
                b_deriv.T[..., np.newaxis] / np.abs(b_deriv).max(), dtype=torch.float32
            ),
            torch.tensor(
                b_deriv_sq.T[..., np.newaxis] / np.abs(b_deriv_sq).max(),
                dtype=torch.float32,
            ),
            torch.tensor(tantan_b.T[..., np.newaxis], dtype=torch.float32),
        ]
    tens_l += [
        torch.tensor(full_b.T[..., np.newaxis], dtype=torch.float32)
    ]  # b field is penultimate column
    if training_data:
        tens_l += [
            torch.tensor(
                full_h.T[..., np.newaxis], dtype=torch.float32
            ),  # target is last column
        ]

    # return ts tensor with shape: (#time steps, #profiles, #features), and scalar tensor with (#profiles, #features)
    return torch.dstack(tens_l), torch.tensor(X.to_numpy(), dtype=torch.float32)


# TODO check report generation for new materials


def main(
    ds=None,
    start_seed=0,
    predict_ploss_directly=False,
    new_materials=False,
    device="cuda:0",
):
    """Main training loop for Residual CNNs

    Args
    ----
    ds: pandas DataFrame
        The dataset
    start_seed: int
        Starting seed, will be incremented by one for each experiment
    predict_ploss_directly: bool
        Adapt the topology to not only predict H field but also power loss P from the H prediction
    new_materials: bool
        Load old 10 materials or the 5 new materials

    Returns
    -------
    logs_d : dict
        Nested dict with experimental results for all seeds and materials
    """
    device = torch.device(device)

    if ds is None:
        if new_materials:
            ds = load_new_materials()
        else:
            ds = pd.read_pickle(PROC_SOURCE / "ten_materials.pkl.gz")
    if DEBUG:
        debug_mats = DEBUG_MATERIALS["new"] if new_materials else DEBUG_MATERIALS["old"]
        ds = pd.concat(
            [
                d.iloc[:10, :]
                for _, d in ds.query("material in @debug_mats").groupby("material")
            ],
            ignore_index=True,
        )

    experiment_uid = str(uuid4())[:5]
    logs_d = {"experiment_uid": experiment_uid, "results_per_material": {}}

    for m_i, (material_lbl, mat_df) in enumerate(ds.groupby("material")):
        mat_df = mat_df.reset_index(drop=True)
        print(f"Train for {material_lbl} (experiment uid: {experiment_uid})")

        mat_df_proc = mat_df.assign(
            kfold=get_stratified_fold_indices(mat_df, K_KFOLD),
        )
        if "material" in mat_df_proc:
            mat_df_proc = mat_df_proc.drop(columns=["material"])

        def run_dyn_training(rep=0):
            # seed
            np.random.seed(rep)
            random.seed(rep)
            torch.manual_seed(rep)

            logs = {
                "loss_trends_train_h": np.full((N_EPOCHS, K_KFOLD), np.nan),
                "loss_trends_val_h": np.full((N_EPOCHS, K_KFOLD), np.nan),
                "loss_trends_train_p": np.full((N_EPOCHS, K_KFOLD), np.nan),
                "loss_trends_val_p": np.full((N_EPOCHS, K_KFOLD), np.nan),
                "model_scripted": [],
                "model_uids": [],
                "start_time": pd.Timestamp.now().round(freq="S"),
                "performance": None,
                "model_size": None,
            }
            # training result container
            results_df = mat_df_proc.loc[:, ["ploss", "kfold"]].assign(
                pred=np.float32(0.0)
            )
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        np.zeros((len(results_df), len(H_PRED_COLS)), dtype=np.float32),
                        columns=H_PRED_COLS,
                    ),
                ],
                axis=1,
            )

            x_cols = [
                c
                for c in mat_df_proc
                if c not in ["ploss", "kfold"] and not c.startswith(("B_t_", "H_t_"))
            ]

            # calculate max elongation for normalization
            b_limit = np.abs(mat_df_proc.loc[:, B_COLS].to_numpy()).max()  # T
            h_limit = min(
                np.abs(mat_df_proc.loc[:, H_COLS].to_numpy()).max(), 150
            )  # A/m

            # normalize on a per-profile base
            b_limit_per_profile = np.abs(mat_df_proc.loc[:, B_COLS].to_numpy()).max(
                axis=1, keepdims=True
            )
            h_limit = h_limit * b_limit_per_profile / b_limit

            for fold_i, (kfold_lbl, test_fold_df) in enumerate(
                mat_df_proc.groupby("kfold")
            ):
                if K_KFOLD > 1:
                    train_fold_df = mat_df_proc.query("kfold != @kfold_lbl")
                else:
                    train_fold_df = mat_df_proc
                train_idx = train_fold_df.index.to_numpy()
                train_fold_df = train_fold_df.reset_index(drop=True).drop(
                    columns="kfold"
                )

                # determine normalizations for training set
                b_limit_fold = b_limit
                b_limit_fold_pp = b_limit_per_profile[train_idx]
                h_limit_fold = h_limit[train_idx]

                if predict_ploss_directly:
                    # store ln ploss mean and std for normalization
                    # TODO: not sure whether standardizing is helpful here
                    # log_ploss = np.log(train_fold_df.ploss.to_numpy())
                    ln_ploss_mean = 0  # np.mean(log_ploss)
                    ln_ploss_std = 1  # np.std(log_ploss)
                else:
                    ln_ploss_mean = 0
                    ln_ploss_std = 1
                train_tensor_ts, train_tensor_scalar = construct_tensor_seq2seq(
                    train_fold_df,
                    x_cols,
                    b_limit_fold,
                    h_limit_fold,
                    b_limit_pp=b_limit_fold_pp,
                    # ln_ploss_mean=ln_ploss_mean,
                    # ln_ploss_std=ln_ploss_std
                )
                n_ts = (
                    train_tensor_ts.shape[-1]
                    - 2  # subtract original B curve and target time series
                )  # number of time series per profile next to B curve
                train_tensor_ts = train_tensor_ts.to(device)
                train_tensor_scalar = train_tensor_scalar.to(device)

                # determine normalizations for validation set
                test_idx = test_fold_df.index.to_numpy()
                b_limit_test_fold = b_limit
                b_limit_test_fold_pp = b_limit_per_profile[test_idx]
                h_limit_test_fold = h_limit[test_idx]

                val_tensor_ts, val_tensor_scalar = construct_tensor_seq2seq(
                    test_fold_df,
                    x_cols,
                    b_limit_test_fold,
                    h_limit_test_fold,
                    b_limit_pp=b_limit_test_fold_pp,
                )
                val_tensor_ts = val_tensor_ts.to(device)
                val_tensor_scalar = val_tensor_scalar.to(device)

                if predict_ploss_directly:
                    # prepare torch tensors for normalization scales
                    b_limit_fold_torch = torch.as_tensor(
                        b_limit_fold, dtype=torch.float32
                    ).to(device)
                    h_limit_fold_torch = torch.as_tensor(
                        h_limit_fold, dtype=torch.float32
                    ).to(device)
                    b_limit_test_fold_torch = torch.as_tensor(
                        b_limit_test_fold, dtype=torch.float32
                    ).to(device)
                    h_limit_test_fold_torch = torch.as_tensor(
                        h_limit_test_fold, dtype=torch.float32
                    ).to(device)
                    freq_scale_torch = torch.as_tensor(
                        FREQ_SCALE, dtype=torch.float32
                    ).to(device)

                # init model
                mdl = TCNWithScalarsAsBias(
                    num_input_scalars=len(x_cols),
                    num_input_ts=1 + n_ts,
                    tcn_layer_cfg=None,
                    scalar_layer_cfg=None,
                )
                loss_h = torch.nn.MSELoss().to(device)
                if predict_ploss_directly:
                    mdl = LossPredictor(mdl)
                    loss_p = torch.nn.MSELoss().to(device)

                opt = torch.optim.NAdam(mdl.parameters(), lr=1e-3)
                pbar = trange(
                    N_EPOCHS,
                    desc=f"Seed {rep}, fold {kfold_lbl}",
                    position=rep * K_KFOLD + kfold_lbl,
                    unit="epoch",
                    mininterval=1.0,
                )
                if rep == 0 and kfold_lbl == 0 and m_i == 0:  # print only once
                    info_input = [
                        torch.ones(
                            (1, 1 + n_ts, len(H_COLS)),
                            dtype=torch.float32,
                        ),
                        torch.ones((1, len(x_cols)), dtype=torch.float32),
                    ]
                    if predict_ploss_directly:
                        info_input += [
                            torch.ones(1, 1, dtype=torch.float32),
                            torch.ones(1, 1, dtype=torch.float32),
                            torch.tensor(1),
                        ]
                    mdl_info = ti_summary(
                        mdl,
                        input_data=info_input,
                        device=device,
                        verbose=0,
                    )
                    pbar.write(str(mdl_info))
                    logs["model_size"] = mdl_info.total_params
                mdl.to(device)

                # generate shuffled indices beforehand
                n_profiles = train_tensor_ts.shape[1]
                idx_mat = []
                for _ in range(N_EPOCHS):
                    idx = np.arange(n_profiles)
                    np.random.shuffle(idx)
                    idx_mat.append(idx)
                idx_mat = np.vstack(idx_mat)

                # Training loop
                val_loss_h = None
                val_loss_p = None
                for i_epoch in pbar:
                    mdl.train()
                    # shuffle profiles
                    indices = idx_mat[i_epoch]
                    train_tensor_ts_shuffled = train_tensor_ts[:, indices, :]
                    train_tensor_scalar_shuffled = train_tensor_scalar[indices, :]

                    if predict_ploss_directly:
                        h_lim_shuffled = h_limit_fold_torch[indices, :]
                        # loss weighting
                        p_loss_weight = i_epoch / N_EPOCHS
                        h_loss_weight = 1.0 - p_loss_weight

                    for i_batch in range(int(np.ceil(n_profiles / BATCH_SIZE))):
                        # extract mini-batch
                        start_marker = i_batch * BATCH_SIZE
                        end_marker = min((i_batch + 1) * BATCH_SIZE, n_profiles)
                        train_tensor_ts_shuffled_n_batched = train_tensor_ts_shuffled[
                            :, start_marker:end_marker, :
                        ]
                        train_tensor_scalar_shuffled_n_batched = (
                            train_tensor_scalar_shuffled[start_marker:end_marker, :]
                        )

                        mdl.zero_grad()
                        # drop target features
                        X_tensor_ts = train_tensor_ts_shuffled_n_batched[
                            :, :, :-1
                        ]  # exclude h field on last column
                        X_tensor_scalar = train_tensor_scalar_shuffled_n_batched[
                            :, :-1
                        ]  # exclude ploss on last column
                        if predict_ploss_directly:
                            # h field as intermediate target
                            h_g_truth = train_tensor_ts_shuffled_n_batched[:, :, [-1]]
                            p_g_truth = train_tensor_scalar_shuffled_n_batched[:, [-1]]
                            output_p, output_h = mdl(
                                X_tensor_ts.permute(1, 2, 0),
                                X_tensor_scalar,
                                b_limit_fold_torch,
                                h_lim_shuffled[start_marker:end_marker],
                                freq_scale_torch,
                            )
                            train_loss_h = loss_h(output_h, h_g_truth)
                            train_loss_p = loss_p(output_p, p_g_truth)
                            loss = (
                                h_loss_weight * train_loss_h
                                + p_loss_weight * train_loss_p
                            )
                            loss.backward()
                            opt.step()
                        else:
                            g_truth = train_tensor_ts_shuffled_n_batched[:, :, [-1]]
                            output_h = mdl(
                                X_tensor_ts.permute(1, 2, 0), X_tensor_scalar
                            ).permute(2, 0, 1)
                            train_loss_h = loss_h(output_h, g_truth)
                            train_loss_h.backward()
                            opt.step()
                    with torch.no_grad():
                        logs["loss_trends_train_h"][
                            i_epoch, fold_i
                        ] = train_loss_h.cpu().item()
                        if predict_ploss_directly:
                            logs["loss_trends_train_p"][
                                i_epoch, fold_i
                            ] = train_loss_p.cpu().item()
                            pbar_str = f"Loss h {train_loss_h.cpu().item():.2e} | val loss h {val_loss_h if val_loss_h is not None else -1.0:.2e} | Loss p {train_loss_p.cpu().item():.2e}"
                        else:
                            pbar_str = f"Loss {train_loss_h.cpu().item():.2e} | val loss {val_loss_h if val_loss_h is not None else -1.0:.2e}"

                    if K_KFOLD > 1:
                        do_validate = i_epoch % 10 == 0 or i_epoch == N_EPOCHS - 1
                    else:
                        do_validate = i_epoch == N_EPOCHS - 1
                    if do_validate:
                        # validation set
                        mdl.eval()
                        with torch.no_grad():
                            if predict_ploss_directly:
                                val_pred_p, val_pred_h = mdl(
                                    val_tensor_ts[:, :, :-1].permute(1, 2, 0),
                                    val_tensor_scalar[:, :-1],
                                    b_limit_test_fold_torch,
                                    h_limit_test_fold_torch,
                                    freq_scale_torch,
                                )
                                val_h_g_truth = val_tensor_ts[:, :, [-1]]
                                val_p_g_truth = val_tensor_scalar[:, [-1]]
                                val_loss_p = (
                                    loss_p(val_pred_p, val_p_g_truth).cpu().item()
                                )
                                val_loss_h = (
                                    loss_h(val_pred_h, val_h_g_truth).cpu().item()
                                )
                                logs["loss_trends_val_p"][
                                    i_epoch, kfold_lbl
                                ] = val_loss_p
                            else:
                                val_pred_h = mdl(
                                    val_tensor_ts[:, :, :-1].permute(1, 2, 0),
                                    val_tensor_scalar[:, :-1],
                                ).permute(2, 0, 1)
                                val_h_g_truth = val_tensor_ts[:, :, [-1]]
                                val_loss_h = (
                                    loss_h(val_pred_h, val_h_g_truth).cpu().item()
                                )
                            logs["loss_trends_val_h"][i_epoch, kfold_lbl] = val_loss_h
                        if np.isnan(val_loss_h):
                            break
                    if val_loss_h is not None:
                        with torch.no_grad():
                            if predict_ploss_directly:
                                pbar_str = f"Loss h {train_loss_h.cpu().item():.2e} | val loss h {val_loss_h if val_loss_h is not None else -1.0:.2e} | Loss p {train_loss_p.cpu().item():.2e}"
                            else:
                                pbar_str = f"Loss {train_loss_h.cpu().item():.2e} | val loss {val_loss_h if val_loss_h is not None else -1.0:.2e}"

                    pbar.set_postfix_str(pbar_str)

                    if half_lr_at is not None:
                        if i_epoch in half_lr_at:
                            for group in opt.param_groups:
                                group["lr"] *= 0.75

                    if i_epoch == N_EPOCHS - 1:  # last epoch
                        with torch.inference_mode():  # take last epoch's model as best model
                            val_tensor_ts_np = val_tensor_ts.cpu().numpy()
                            val_tensor_scalars_np = val_tensor_scalar.cpu().numpy()
                            h_pred_val_np = (
                                val_pred_h.squeeze().cpu().numpy().T * h_limit_test_fold
                            ).astype(np.float32)
                            if predict_ploss_directly:
                                results_df.loc[
                                    results_df.kfold == kfold_lbl, "pred"
                                ] = np.exp(
                                    val_pred_p.cpu().numpy() * ln_ploss_std
                                    + ln_ploss_mean
                                ).astype(
                                    np.float32
                                )
                            else:
                                results_df.loc[
                                    results_df.kfold == kfold_lbl, "pred"
                                ] = get_bh_integral_from_two_mats(
                                    freq=np.exp(
                                        val_tensor_scalars_np[:, 0].reshape(-1, 1)
                                    )
                                    * FREQ_SCALE,
                                    b=val_tensor_ts_np[:, :, -2].T * b_limit_test_fold,
                                    h=h_pred_val_np,
                                ).astype(
                                    np.float32
                                )
                            results_df.loc[
                                results_df.kfold == kfold_lbl,
                                [c for c in results_df if c.startswith("h_pred_")],
                            ] = h_pred_val_np
                # end of fold
                logs["model_scripted"].append(torch.jit.script(mdl.cpu()))
                logs["model_uids"].append(str(uuid4())[:8])

            # for further book keeping
            logs["performance"] = calculate_metrics(
                results_df.loc[:, "pred"], results_df.loc[:, "ploss"]
            )
            logs["results_df"] = results_df
            return logs

        n_seeds = N_SEEDS
        print(f"Parallelize over {n_seeds} seeds with {N_JOBS} processes..")
        # start experiments in parallel processes
        # list of dicts
        # mat_log = prll(delayed(run_dyn_training)(s) for s in range(start_seed, n_seeds + start_seed))
        # logs_d[material_lbl] = {'performance': pd.DataFrame.from_dict([m['performance'] for m in mat_log]),
        #                        'misc': [m for m in mat_log]}

        # Note that parallel processes won't work in conjunction with a GPU (memory won't be released with Pytorch)
        logs_d["results_per_material"][material_lbl] = [
            run_dyn_training(i) for i in range(start_seed, n_seeds + start_seed)
        ]

    return logs_d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN Training")
    parser.add_argument(
        "-t", "--tag", default="", help="A tag/comment describing the experiment."
    )
    parser.add_argument(
        "-g", "--gpu", default="0", help="GPU device to use. -1 for CPU."
    )
    args = parser.parse_args()
    device_str = f"cuda:{args.gpu}" if int(args.gpu) >= 0 else f"cpu"
    # load data set and featurize
    if TRAIN_ON_NEW_MATERIALS:
        ds = load_new_materials()
    else:
        ds = pd.read_pickle(PROC_SOURCE / "ten_materials.pkl.gz")
    # Feature Engineering
    ds = engineer_features(ds, with_b_sat=not TRAIN_ON_NEW_MATERIALS)

    # Training main loop
    logs = main(
        ds=ds,
        predict_ploss_directly=DO_PREDICT_P_DIRECTLY,
        new_materials=TRAIN_ON_NEW_MATERIALS,
        device=device_str,
    )
    # dump results to files
    bookkeeping(
        logs,
        debug=DEBUG,
        experiment_info={
            "subsample_factor": SUBSAMPLE_FACTOR,
            "batch_size": BATCH_SIZE,
            "n_folds": K_KFOLD,
            "predicts_p_directly": DO_PREDICT_P_DIRECTLY,
            "n_epochs": N_EPOCHS,
            "tag": args.tag,
        },
    )
