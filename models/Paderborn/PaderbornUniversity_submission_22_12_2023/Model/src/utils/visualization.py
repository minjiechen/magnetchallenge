import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.experiments import get_bh_integral_from_two_mats


plt.style.use("default")
plt.rcParams.update(
    {
        "figure.dpi": 120,  # renders images larger for notebook
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "serif",
    }
)


def visualize_rel_error_report(preds, gtruth, 
                               title="MagNet Challenge Pretest Results for 10 Known Materials - Due 11/01/2023\nTeam Paderborn University"):
    """Render a visual report as requested by the MagNet Challenge hosts, see
    (https://github.com/minjiechen/magnetchallenge/blob/main/pretest/PretestResultsPDF.pdf).
    Argument preds are the model predictions, gtruth the ground truth of the power losses.
    Both arguments must be either dictionaries or pandas DataFrames, in order to
    distinguish between materials. Moreover, both should be likewise sorted.
    Both, preds and gtruth, must contain they key/column 'material', which will be filtered.
    If preds is a power loss prediction, it should only contain two columns, 'ploss' and 'material'.
    If preds is an H prediction, it should contain the sequences with 'h_pred_xx' columns and
    no 'ploss' column."""

    if isinstance(preds, dict):
        preds = pd.DataFrame(preds)
    if isinstance(gtruth, dict):
        gtruth = pd.DataFrame(gtruth)
    assert "material" in preds, 'Column "material" missing in preds argument'
    assert "material" in gtruth, 'Column "material" missing in gtruth argument'

    is_h_prediction = "ploss" not in preds

    n_materials = gtruth.groupby("material").ngroups
    fig, axes = plt.subplots(
        nrows=np.ceil(n_materials / 2).astype(int),
        ncols=2,
        sharex=False,
        sharey=False,
        figsize=(8.26, 11.69),
    )
    # joined_df = pd.concat([gtruth.reset_index(drop=True), preds.reset_index(drop=True).drop(columns=['material'])], axis=1)
    for (m_lbl, preds_mat_df), ax in zip(preds.groupby("material"), axes.flatten()):
        gtruth_mat = gtruth.query("material == @m_lbl")
        if is_h_prediction:
            preds_mat = preds_mat_df.loc[
                :, [c for c in preds_mat_df if c.startswith("h_pred")]
            ].to_numpy()

            p_pred = get_bh_integral_from_two_mats(
                freq=gtruth_mat.freq.to_numpy(),
                b=gtruth_mat.loc[
                    :, [c for c in gtruth_mat if c.startswith("B_t_")]
                ].to_numpy()[:, :: 1024 // preds_mat.shape[1]],
                h=preds_mat,
            )
        else:
            # p prediction was given
            p_pred = preds_mat_df.loc[:, 'ploss']
        assert (
            p_pred.shape[0] == gtruth_mat.ploss.shape[0]
        ), f"shapes mismatch, preds {m_lbl} != gtruth {gtruth.loc[gtruth.material == str(m_lbl), :].material.unique()[0]}, with {p_pred.shape=} != {gtruth_mat.shape=}"
        err = (np.abs(p_pred - gtruth_mat.ploss) / gtruth_mat.ploss * 100).ravel()
        avg = err.mean()
        percentile_95 = np.percentile(err, 95)
        percentile_99 = np.percentile(err, 99)
        err_max = err.max()
        ax.hist(err, bins=50, density=True)
        ax.set_ylabel("Ratio of Data Points")
        ax.set_xlabel("Relative Error of Core Loss [%]")
        ax.set_title(
            f"Error Distribution for {m_lbl}\nAvg={avg:.3f}%, 95-Prct={percentile_95:.1f}%, 99-Prct={percentile_99:.1f}%, Max={err_max:.2f}%",
            fontsize=8,
        )
        for mark, annot, height in zip(
            [avg, percentile_95, err_max],
            [f"Avg={avg:.3f}%", f"95-Prct={percentile_95:.2f}%", f"Max={err_max:.2f}%"],
            [0.9, 0.6, 0.2],
        ):
            vline = ax.axvline(
                mark,
                ymin=0,
                ymax=height,
                c="tab:red",
                ls="dashed",
            )
            x, y = vline.get_xydata()[-1]
            y = y * ax.get_ylim()[-1]
            xtext = x + 0.01 * ax.get_xlim()[-1] * (-1 if height == 0.2 else 1)
            ax.annotate(
                annot,
                xy=[x, y],
                xytext=[xtext, y],
                color="tab:red",
                verticalalignment="top",
                horizontalalignment="right" if height == 0.2 else "left",
                fontsize=8,
            )
    fig.suptitle(
        title,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
