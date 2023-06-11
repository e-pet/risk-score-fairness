import numpy as np
import scipy.interpolate
import scipy.signal
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.isotonic import IsotonicRegression
from warnings import warn
from prg import create_prg_curve
from calibration import get_equal_mass_bins


def get_confusion_matrix(target, preds, threshold=0.5, reduce="micro"):
    """Calculate the number of tp, fp, tn, fn.

    Args:
        target:
            An ``(N, C)`` or ``(N, C, X)`` array of true labels (0 or 1)    
        preds:
            An ``(N, C)`` or ``(N, C, X)`` array of predictions (0 or 1), logits, or probabilities
        threshold:
            Float decision threshold to use if preds are not already labels, between -1 and 1.
        reduce:
            One of ``'micro'``, ``'macro'``, ``'samples'``

    Return:
        Returns tp, fp, tn, fn.
        The shape of the returned arrays depends on the shape of the inputs
        and the ``reduce`` parameter:

        If inputs are of the shape ``(N, C)``, then
        - If ``reduce='micro'``, the returned arrays are 1 element arrays
        - If ``reduce='macro'``, the returned arrays are ``(C,)`` arrays
        - If ``reduce'samples'``, the returned arrays are ``(N,)`` arrays

        If inputs are of the shape ``(N, C, X)``, then
        - If ``reduce='micro'``, the returned arrays are ``(N,)`` arrays
        - If ``reduce='macro'``, the returned arrays are ``(N,C)`` arrays
        - If ``reduce='samples'``, the returned arrays are ``(N,X)`` arrays

    ADAPTED FROM TORCHMETRICS
    """
    if target.ndim == 1:
        target = target.reshape(1, -1)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)

    dim = 1  # for "samples"
    if reduce == "micro":
        dim = (0, 1) if preds.ndim == 2 else (1, 2)
    elif reduce == "macro":
        dim = 0 if preds.ndim == 2 else 2

    assert(-1 <= threshold <= 1)
    if (preds < 0).any() and threshold == 0.5:
        warn("preds seem to be logits, but using default decision threshold of 0.5. Consider adjusting the threshold.")
    
    if not issubclass(preds.dtype.type, np.integer):
        # We're dealing with logits or probabilities that still need to be turned into labels
        preds = (preds > threshold).astype(int)

    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    tp = (true_pred * pos_pred).sum(axis=dim)
    fp = (false_pred * pos_pred).sum(axis=dim)

    tn = (true_pred * neg_pred).sum(axis=dim)
    fn = (false_pred * neg_pred).sum(axis=dim)

    return tp, fp, tn, fn


def _ratio(numerator, denominator):
    if denominator == 0:
        return np.nan
    else:
        return numerator / denominator


def accuracy(target, preds, threshold=0.5):
    N = len(preds)
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TP + TN, N)


def recall(target, preds, threshold=0.5):
    # =TPR=sensitivity
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TP, TP + FN)


def precision(target, preds, threshold=0.5):
    # =PPV
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TP, TP + FP)


def f1_score(target, preds, threshold=0.5):
    # harmonic mean of precision and recall
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TP, TP + 0.5 * (FP+FN))


def specificity(target, preds, threshold=0.5):
    # =TNR
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TN, TN + FP)


def selection_rate(target, preds, threshold=0.5):
    N = len(preds)
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TP + FP, N)


def avg_underrepr(yhat_test, group_mask_test, y_all=None, group_mask_all=None, risk_quantile_bounds=[0.5, 1.0],
                  return_datapoints=False, target_repr=None):
    if target_repr is None:
        # get the target representation rate: what is the fraction of people with y=1 that are from the group?
        target_repr = sum(y_all[group_mask_all]) / sum(y_all)
    assert 0 <= target_repr <= 1

    unique_scores, val_counts = np.unique(yhat_test, return_counts=True)
    if len(unique_scores) < 100:
        threshs = unique_scores
    else:
        threshs, val_counts = get_equal_mass_bins(yhat_test, num_bins=100, return_counts=True)

    if risk_quantile_bounds is not None:
        lower_idx = round((len(threshs)-1)*risk_quantile_bounds[0])
        upper_idx = round((len(threshs)-1)*risk_quantile_bounds[1])
        threshs = threshs[lower_idx:upper_idx]
        val_counts = val_counts[lower_idx:upper_idx]

    observed_repr = np.zeros((len(threshs),))

    for idx, thresh in enumerate(threshs):
        pred_labels = yhat_test >= thresh
        if sum(pred_labels) > 0:
            observed_repr[idx] = sum(pred_labels[group_mask_test]) / sum(pred_labels)
        else:
            observed_repr[idx] = np.nan

    rel_repr = observed_repr / target_repr
    rel_under_repr = np.minimum(rel_repr, 1.)
    # This is an average / expectation taken over the _risk score distribution in the test set_!
    nan_mask = np.isnan(rel_under_repr)
    assert nan_mask.sum() < len(rel_under_repr)
    avg_under_repr = np.sum((1 - rel_under_repr[~nan_mask]) * val_counts[~nan_mask]) / val_counts[~nan_mask].sum()
    assert isinstance(avg_under_repr, float)
    assert 0 <= avg_under_repr <= 1

    if return_datapoints:
        assert len(threshs) == len(rel_under_repr)
        return avg_under_repr, threshs, rel_repr
    else:
        return avg_under_repr


def calibration_refinement_error(target, pred_probs, num_bins=15, adaptive=True, abort_if_not_monotonic=False):
    # adapted from on https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
    # This returns a) a calibration error / loss estimate, and b) a refinement loss estimate,
    # following the proper scoring rule decomposition presented in, e.g.,
    # Kull and Flach (2015), "Novel Decompositions of Proper Scoring Rules for Classification: Score Adjustment as
    #                         Precursor to Calibration."
    # Either adaptive or static binning can be used.
    # The first term is exactly the typical calibration error. With static binning, this is called ECE; with adaptive
    # binning ACE. It is, however, NOT identical to the calibration loss term in the BS decomposition, because that uses

    # implemented and tested for the binary case. might also work for multi-class, not sure
    assert len(np.unique(target) <= 2)
    assert issubclass(target.dtype.type, np.integer)
    assert np.issubdtype(pred_probs.dtype, np.floating)
    assert np.all(target.shape == pred_probs.shape)
    assert np.all(0 <= pred_probs) and np.all(pred_probs <= 1)

    if num_bins == 'sweep':
        # find the number of bins automatically by searching for the maximum number of bins for which bin incidences
        # are still monotonic, starting from a default of 15.
        # Inspired by this paper (but we're using a more efficient search procedure):
        # https://home.cs.colorado.edu/~mozer/Research/Selected%20Publications/reprints/Roelofsetal2022.pdf
        curr_num_bins = 15
        upper_bound = round(len(target)/2)
        lower_bound = 1
        print('Starting bin count sweep...')
        while not lower_bound == upper_bound == curr_num_bins:
            print(f'nbin={curr_num_bins}')
            try:
                ece, ref_err = calibration_refinement_error(target, pred_probs, num_bins=curr_num_bins,
                                                            adaptive=adaptive, abort_if_not_monotonic=True)
            except ValueError:
                upper_bound = curr_num_bins - 1
                curr_num_bins = round(lower_bound + (upper_bound - lower_bound)/2)
            else:
                lower_bound = curr_num_bins
                curr_num_bins = min(curr_num_bins * 2, round(lower_bound + (upper_bound - lower_bound)/2))
            curr_num_bins = min(upper_bound, max(lower_bound+1, curr_num_bins))

        if curr_num_bins == 1:
            ece, ref_err = calibration_refinement_error(target, pred_probs, num_bins=curr_num_bins, adaptive=adaptive)
        print(f'Final nbin={curr_num_bins}')
        return ece, ref_err

    else:

        b = np.linspace(start=0, stop=1.0, num=num_bins)

        if adaptive:
            # compute adaptive calibration error, ACE
            # See Nixon, M. Dusenberry et al., Measuring Calibration in Deep Learning, 2020.
            b = np.quantile(pred_probs, b)
            b = np.unique(b)
            num_bins = len(b)
        else:
            # compute expected calibration error, ECE
            pass

        bins = np.digitize(pred_probs, bins=b, right=True)
        incidences = np.zeros((num_bins, ))
        mean_abs_calib_error = 0
        refinement_error = 0
        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                mean_pred = np.mean(pred_probs[mask])
                incidences[b] = np.mean(target[mask])
                if b > 0 and incidences[b] < incidences[b-1] and abort_if_not_monotonic:
                    raise ValueError
                mean_abs_calib_error += np.sum(mask) * np.abs(mean_pred - incidences[b])
                refinement_error += np.sum(mask) * incidences[b] * (1 - incidences[b])

        return mean_abs_calib_error / pred_probs.shape[0], refinement_error / pred_probs.shape[0]


def ece(target, pred_probs, method, ci_alpha=None, n_samples=100, num_bins=15):
    if method == 'static':
        ece = calibration_refinement_error(target, pred_probs, num_bins=num_bins, adaptive=False)[0]
        return ece
    elif method == 'adaptive':
        ece = calibration_refinement_error(target, pred_probs, num_bins=num_bins, adaptive=True)[0]
        return ece
    if method == 'IR':
        probs = isotonic_calibration(target, pred_probs)
        ece = np.mean(np.abs(pred_probs - probs))
        return ece
    elif method == 'loess':
        probs, probs_samples = loess_calibration(target, pred_probs, n_samples=n_samples)
        ece_arr = np.nanmean(np.abs(pred_probs[:, None] - probs_samples), axis=0)
        ece = np.mean(np.abs(pred_probs - probs))
        ci_lower, ci_upper = np.nanquantile(ece_arr, q=[ci_alpha, 1-ci_alpha])
        return ece, ci_lower, ci_upper
    else:
        raise NotImplementedError


def loess_calibration(target, pred_probs, n_samples=None, xvals=None):
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    # it: The number of residual-based reweightings to perform.
    # frac: Between 0 and 1. The fraction of the data used when estimating each y-value.
    # delta: Distance within which to use linear-interpolation instead of weighted regression.
    if xvals is None:
        delta = 0.005
    else:
        delta = 0

    # Austin and Steyerberg (2013) say frac=0.75 (="span" in their terminology) and it=0 are good for calibration
    # analyses. https://onlinelibrary.wiley.com/doi/full/10.1002/sim.5941
    # However, this frac=0.75 smoothes very strongly and is basically incapable of returning "sharp" calibration curves.
    # Hence, let me try to choose an appropriate value for frac based on the sample size.
    # (frac: fraction of samples that is taken into account for regressing at a given x value, inversely weighted by
    # distance.)
    # I want the number of datapoint taken into account to alway be ~250.
    frac = max(0.3, min(1.0, 250/len(target)))
    calib_probs = sm_lowess(target, pred_probs, frac=frac, it=0, return_sorted=False)

    if n_samples is not None:
        # Reference for bootstrapping CIs for loess-based calibration:
        # https://onlinelibrary.wiley.com/doi/10.1002/sim.6167
        N_predictions = len(target)

        if xvals is None:
            calib_probs_samples = np.zeros((N_predictions, n_samples))
        else:
            calib_probs_samples = np.zeros((len(xvals), n_samples))

        for idx in range(n_samples):
            bs_idces = np.random.choice(range(N_predictions), N_predictions)
            target_bs = target[bs_idces]
            pred_probs_bs = pred_probs[bs_idces]
            calib_probs_samples[:, idx] = sm_lowess(target_bs, pred_probs_bs, frac=frac, it=0,
                                                    xvals=xvals, return_sorted=False, delta=delta)
        return calib_probs, calib_probs_samples
    else:
        return calib_probs


def isotonic_calibration(target, pred_probs, n_boot=0, rng=None):
    isotonic = IsotonicRegression(y_min=0, y_max=1)
    isotonic.fit(pred_probs, target)
    isotonic_probs = isotonic.predict(pred_probs)

    if n_boot > 0:
        isotonic_probs_boots = np.zeros((len(isotonic_probs), n_boot))
        boot_isotonic_probs = np.zeros((len(isotonic_probs), n_boot))
        boot_pred_probs = np.zeros((len(isotonic_probs), n_boot))
        boot_target = np.zeros((len(isotonic_probs), n_boot))

        if rng is None:
            rng = np.random.RandomState()

        for ii in range(0, n_boot):
            # One (symptom of an underlying) problem with this resampling method is that it will assign 0 uncertainty to
            # regions where the initial isotonic regression identifies a bin with conditional event likelihood 0.
            # That seems very unreasonable. Is there any way to fix that?
            # (This is exactly the method Dimitriadis et al. (2021) describe for their UQ. Their plots also seem to
            # suffer from the same problem.
            # resample predictions
            boot_idces = rng.randint(isotonic_probs.shape[0], size=len(isotonic_probs))
            boot_pred_probs[:, ii] = pred_probs[boot_idces]
            # resample targets assuming initial isotonic regression to be correct
            boot_target[:, ii] = rng.binomial(1, p=isotonic_probs[boot_idces])
            # fit another isotonic regression to the resampled data
            boot_isotonic_probs[:, ii] = isotonic_calibration(boot_target[:, ii], boot_pred_probs[:, ii], n_boot=0)
            isotonic_probs_boots[:, ii] = scipy.interpolate.interp1d(boot_pred_probs[:, ii], boot_isotonic_probs[:, ii],
                                                                     kind="nearest",
                                                                     fill_value="extrapolate")(pred_probs)
            assert np.all(isotonic_probs_boots[:, ii] <= 1)

        return isotonic_probs, isotonic_probs_boots
    else:
        return isotonic_probs


def bootstrap_roc_curve(target, pred_probs, num_bootstraps):
    assert len(target) == len(pred_probs)
    N_predictions = len(target)
    fpr = np.arange(0, 1+1e-7, 0.05)
    tpr_bs = np.zeros((num_bootstraps, len(fpr)))

    for bs_idx in range(num_bootstraps):
        bs_idces = np.random.choice(range(N_predictions), N_predictions)
        fpr_loc, tpr_loc, _ = roc_curve(target[bs_idces], pred_probs[bs_idces])
        fpr_loc_unq = np.unique(fpr_loc)
        tpr_loc_unq = np.zeros_like(fpr_loc_unq)
        for idx, fpr_val in enumerate(fpr_loc_unq):
            tpr_loc_unq[idx] = tpr_loc[fpr_loc == fpr_val].max()
        assert np.all(np.diff(fpr_loc_unq) > 0)  # must be increasing for the interpolation below to work
        tpr_bs[bs_idx, :] = np.interp(fpr, fpr_loc_unq, tpr_loc_unq, left=0.0, right=1.0)

    return fpr, tpr_bs


def bootstrap_prg_curve(target, pred_probs, num_bootstraps):
    assert len(target) == len(pred_probs)
    N_predictions = len(target)
    recall_gains = np.arange(0, 1+1e-7, 0.05)
    precision_gains_bs = np.zeros((num_bootstraps, len(recall_gains)))

    for bs_idx in range(num_bootstraps):
        bs_idces = np.random.choice(range(N_predictions), N_predictions)
        prg_curve = create_prg_curve(target[bs_idces], pred_probs[bs_idces])
        recall_gains_loc = prg_curve['recall_gain']
        precision_gains_loc = prg_curve['precision_gain']
        recall_gains_loc_unq = np.unique(recall_gains_loc)
        precision_gains_loc_unq = np.zeros_like(recall_gains_loc_unq)
        for idx, recall_gain_val in enumerate(recall_gains_loc_unq):
            precision_gains_loc_unq[idx] = precision_gains_loc[recall_gains_loc == recall_gain_val].max()
        assert np.all(np.diff(recall_gains_loc_unq) > 0)  # must be increasing for the interpolation below to work
        precision_gains_bs[bs_idx, :] = np.interp(recall_gains, recall_gains_loc_unq, precision_gains_loc_unq)

    return recall_gains, precision_gains_bs


def auroc(target, preds):
    if sum(target) > 0 and sum(target == 0) > 0:
        # There are both positive and negative examples
        r = roc_auc_score(target, preds)
    else: 
        r = np.nan

    return r
