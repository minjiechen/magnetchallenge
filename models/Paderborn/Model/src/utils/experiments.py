import numpy as np
import pandas as pd
from utils.data import ALL_B_COLS, ALL_H_COLS

# bsat map
BSAT_MAP = {
    "3C90": 0.47,
    "3C94": 0.47,
    "3E6": 0.46,
    "3F4": 0.41,
    "77": 0.51,
    "78": 0.48,
    "N27": 0.50,
    "N30": 0.38,
    "N49": 0.49,
    "N87": 0.49,
    #TODO what are new values for new materials?
    'A': 1,
    'B': 1,
    'C': 1,
    'D': 1,
    'E': 1,
}


def get_bh_integral(df):
    """Given the B and H curve as well as the frequency in the pandas DataFrame df,
      calculate the area within the polygon"""
    # offset polygon into first quadrant
    b, h = (
        df.loc[:, ALL_B_COLS].to_numpy() + 0.5,  # T
        df.loc[:, ALL_H_COLS].to_numpy() + 1000,  # A/m
    )
    return (
        df.freq
        * 0.5
        * np.abs(np.sum(b * (np.roll(h, 1, axis=1) - np.roll(h, -1, axis=1)), axis=1))
    )  # shoelace formula

def get_bh_integral_from_two_mats(freq, b, h):
    """ b and h are numpy matrices with shape (#samples, #timesteps)"""
    # offset b and h into first quadrant
    h_with_offset = h + 1000 # A/m
    b_with_offset = b + 0.5 # T
    return freq.ravel() * 0.5 * np.abs(
        np.sum(b_with_offset * 
               (np.roll(h_with_offset, 1, axis=1) - np.roll(h_with_offset, -1, axis=1)),
                axis=1))

def get_stratified_fold_indices(df, n_folds):
    """Given a Pandas Dataframe df, return a Pandas Series with the kfold labels for the test set.
    The test set labels are distributed such that they are stratified along the B-field's peak-2-peak value.
    The labels are an enumeration of the test sets, e.g., for a 4-fold training each row will be labeled in [0, 3].

    Example:
    --------
    >>> ds = pd.read_pickle(PROC_SOURCE / "ten_materials.pkl.gz")
    >>> kfold_lbls = get_stratified_fold_indices(ds.query("material == '3F4'"), 4)  # 4-fold
    >>> kfold_lbls
    0       0
    1       2
    2       0
    3       2
    4       2
        ..
    6558    1
    6559    0
    6560    0
    6561    1
    6562    3
    Name: kfold_tst_lbl, Length: 6563, dtype: int64
    >>> ds.loc[:, 'kfold'] = kfold_lbls
    >>> for i in range(4):
    ...     test_ds = ds.query("kfold == @i")
    ...     train_ds = ds.query("kfold != @i")
            # train on this fold
    """
    full_b = df.loc[:, [f"B_t_{k}" for k in range(1024)]].to_numpy()

    df = (
        df.assign(b_peak2peak=full_b.max(axis=1) - full_b.min(axis=1))
        .reset_index(drop=False, names="orig_idx")
        .sort_values("b_peak2peak", ascending=True)
        .assign(
            kfold_tst_lbl=np.tile(np.arange(n_folds), np.ceil(len(df) / n_folds).astype(int))[: len(df)])
        .sort_values("orig_idx", ascending=True)
    )
    return df.kfold_tst_lbl


def form_factor(x):
    """
    definition:      kf = rms(x) / mean(abs(x))
    for ideal sine:  np.pi/(2*np.sqrt(2))
    """
    return np.sqrt(np.mean(x**2, axis=1)) / np.mean(np.abs(x), axis=1)


def crest_factor(x):
    """
    definition:      kc = rms(x) / max(x)
    for ideal sine:  np.sqrt(2)
    """
    return np.max(np.abs(x), axis=1) / np.sqrt(np.mean(x**2, axis=1))



def bool_filter_sine(b, rel_kf=0.01, rel_kc=0.01, rel_0_dev=0.1):
    """
    b: input flux density (nxm)-array with n m-dimensional flux density waveforms
    rel_kf: (allowed) relative deviation of the form factor for sine classification
    rel_kc: (allowed) relative deviation of the crest factor for sine classification
    rel_0_dev: (allowed) relative deviation of the first value from zero (normalized on the peak value)
    """
    kf_sine = np.pi / (2 * np.sqrt(2))
    kc_sine = np.sqrt(2)

    filter_bool = [True] * b.shape[0]

    statements = [
        list(form_factor(b) < kf_sine * (1 + rel_kf)),  # form factor based checking
        list(form_factor(b) > kf_sine * (1 - rel_kf)),  # form factor based checking
        list(crest_factor(b) < kc_sine * (1 + rel_kc)), # crest factor based checking
        list(crest_factor(b) > kc_sine * (1 - rel_kc)), # crest factor based checking
        list(b[:, 0] < np.max(b, axis=1) * rel_0_dev),  # starting value based checking
        list(b[:, 0] > -np.max(b, axis=1) * rel_0_dev), # starting value based checking
    ]

    for statement in statements:
        filter_bool = [a and zr for a, zr in zip(filter_bool, statement)]

    

    return filter_bool

def bool_filter_triangular(b, rel_kf=0.005, rel_kc=0.005):
    kf_triangular = 2/np.sqrt(3)
    kc_triangular = np.sqrt(3)

    filter_bool = [True] * b.shape[0]

    statements = [list(form_factor(b) < kf_triangular * (1 + rel_kf)),
                  list(form_factor(b) > kf_triangular * (1 - rel_kf)),
                  list(crest_factor(b) < kc_triangular * (1 + rel_kc)),
                  list(crest_factor(b) > kc_triangular * (1 - rel_kc))]

    for statement in statements:
        filter_bool = [a and zr for a, zr in zip(filter_bool, statement)]

    return filter_bool

def get_waveform_est(full_b):
    """From Till's tp-1.4.7.3.1 NB, return waveform class.
    Postprocessing from wk-1.1-EDA NB."""
  
    # labels init all with 'other'
    k = np.zeros(full_b.shape[0], dtype=int)
    
    # square
    k[np.all(np.abs(full_b[:, 250:500:50] - full_b[:, 200:450:50]) / np.max(np.abs(full_b), axis=1, keepdims=True) < 0.05, axis=1) & np.all(full_b[:, -200:]< 0, axis=1)] = 1
    
    # triangular
    k[bool_filter_triangular(full_b, rel_kf=0.01, rel_kc=0.01)] = 2

    # sine
    k[bool_filter_sine(full_b, rel_kf=0.01, rel_kc=0.01)] = 3

    # postprocess "other" signals in frequency-domain, to recover some more squares, triangles, and sines
    n_subsample = 32
    other_b = full_b[k == 0, ::n_subsample]
    other_b /= np.abs(other_b).max(axis=1, keepdims=True)
    other_b_ft = np.abs(np.fft.fft(other_b, axis=1))
    other_b_ft /= other_b_ft.max(axis=1, keepdims=True)
    msk_of_newly_identified_sines = np.all((other_b_ft[:, 3:10] < 0.03) & (other_b_ft[:, [2]] < 0.2), axis=1)
    msk_of_newly_identified_triangs = np.all(((other_b_ft[:, 1:8] - other_b_ft[:, 2:9]) > 0), axis=1) | np.all(((other_b_ft[:, 1:8:2] > 1e-2) & (other_b_ft[:, 2:9:2] < 1e-2)), axis=1)
    msk_of_newly_identified_triangs = msk_of_newly_identified_triangs & ~msk_of_newly_identified_sines
    msk_of_newly_identified_squares = np.all((other_b_ft[:, 1:4:2] > 1e-2) & (other_b_ft[:, 2:5:2] < 1e-3), axis=1)
    msk_of_newly_identified_squares = msk_of_newly_identified_squares & ~msk_of_newly_identified_sines & ~msk_of_newly_identified_triangs
    idx_sines = np.arange(k.size)[k == 0][msk_of_newly_identified_sines]
    idx_triangs = np.arange(k.size)[k == 0][msk_of_newly_identified_triangs]
    idx_squares = np.arange(k.size)[k == 0][msk_of_newly_identified_squares]
    k[idx_squares] = 1
    k[idx_triangs] = 2
    k[idx_sines] = 3
    return k

def engineer_features(ds, with_b_sat=False):
    """Add features to data set"""

    full_b = ds.loc[:, ALL_B_COLS].to_numpy()
    waveforms = get_waveform_est(full_b)
    ds = pd.concat(
        [
            ds,
            pd.get_dummies(waveforms, prefix="wav", dtype=float).rename(
                columns={
                    "wav_0": "wav_other",
                    "wav_1": "wav_square",
                    "wav_2": "wav_triangular",
                    "wav_3": "wav_sine",
                }
            ),
        ],
        axis=1,
    )
    
    dbdt = full_b[:, 1:] - full_b[:, :-1]
    b_peak2peak = full_b.max(axis=1) - full_b.min(axis=1)
    # fft features (experimental)
    #b_ft = np.abs(np.fft.fft(full_b[:, ::32], axis=1))

    ds = ds.assign(
        b_peak2peak=b_peak2peak,
        log_peak2peak=np.log(b_peak2peak),
        mean_abs_dbdt=np.mean(np.abs(dbdt), axis=1),
        log_mean_abs_dbdt=np.log(np.mean(np.abs(dbdt), axis=1)),
        sample_time=1 / ds.loc[:, "freq"],
        #**{f'ft_{k}': b_ft[:, k] for k in range(1, 10)}
    )
    if with_b_sat:
        ds = ds.assign(db_bsat=b_peak2peak / ds.material.map(BSAT_MAP))
    return ds

