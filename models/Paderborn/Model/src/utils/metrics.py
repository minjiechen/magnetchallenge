import numpy as np


def calculate_metrics(pred, gtruth):
    """Given predictions and the corresponding ground truth in unnormalized
    quantities, return a set of metrics as dictionary.
    Assuming pred and gtruth are array-like, and have either dim=1, or, if more, each
    column is to be treated equally.

    Example
    --------
        >>> calculate_metrics(5+np.random.uniform(size=100), 5+np.random.uniform(size=100))
        {'mse': 0.18452681979466903,
        'mae': 0.359136744337167,
        'avg-abs-rel-err': 0.06506427064735477,
        'percentile_5_rel_err': 0.004049281460581758,
        'percentile_95_rel_err': 0.1395315354033197,
        'l_infty': 0.9475492352105599,
        'l_infty_over': 0.8902558869211452,
        'l_infty_under': -0.9475492352105599}

    """

    if hasattr(pred, "values") or hasattr(pred, "to_numpy"):
        pred = pred.to_numpy()
    if hasattr(gtruth, "values") or hasattr(gtruth, "to_numpy"):
        gtruth = gtruth.to_numpy()

    assert (
        pred.shape == gtruth.shape
    ), f"pred and gtruth shape do not match ({pred.shape} != {gtruth.shape})"

    diffs = pred - gtruth
    rel_err = np.abs(diffs) / np.abs(gtruth)
    return {
        "mse": np.mean(diffs**2),
        "mae": np.mean(np.abs(diffs)),
        "avg-abs-rel-err": np.mean(
            rel_err
        ),  # taking the mean according to webinar-3 page 9
        "percentile_95_rel_err": np.percentile(rel_err, 95),
        "percentile_99_rel_err": np.percentile(rel_err, 99),
        "l_infty": np.max(np.abs(diffs)),
        "l_infty_over": np.max(diffs),
        "l_infty_under": np.min(diffs),
    }
