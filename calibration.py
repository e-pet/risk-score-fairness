import numpy as np
from typing import List, Tuple, TypeVar
from betacal import BetaCalibration
from sklearn.base import BaseEstimator, RegressorMixin

Data = List[Tuple[float, float]]  # List of (predicted_probability, true_label).
Bins = List[float]  # List of bin boundaries, excluding 0.0, but including 1.0.
BinnedData = List[Data]  # binned_data[i] contains the data in bin i.
T = TypeVar('T')

eps = 1e-6

VERBOSE = False


def get_equal_mass_bins(probs, num_bins, min_vals_per_bin=None, return_counts=False):
    """Get bins that contain approximately an equal number of data points."""
    # The original implementation in https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py
    # has some issues with highly repetitive values. Sometimes the method yields highly imbalanced bins, with some
    # being empty and others containing many values. This is my attempt at doing better.

    # This is basic quantile binning, a standard practical approach towards equal mass / equal frequency binning.
    # Notice, however, that it can "break" in the case of many identical values, because then successive quantiles can
    # coincide at the same value. We will try to deal with that further below.
    # (See https://blogs.sas.com/content/iml/2014/11/05/binning-quantiles-rounded-data.html for a simple example of this
    # issue, which in the present context will typically arise when using tree models.)
    # The "+1e-7" is because due to floating point inaccuracies, otherwise sometimes a final "1.0" bin boundary is
    # created and sometimes not. Here, we apparently want it. (See above.)
    # I did not investigate in detail, but the "median_unbiased" method seemed to "work better" with a lot of repetitive
    # values. (It's also the recommended method as per the documentation.)
    bin_arr = np.quantile(probs, np.arange(1/num_bins, 1+1e-7, 1/num_bins), method="median_unbiased")

    assert len(bin_arr) == num_bins

    bin_idces = np.searchsorted(bin_arr, probs)
    assert bin_idces.max() <= (num_bins - 1)
    assert bin_idces.min() >= 0

    _, bin_num_vals_counts = np.unique(bin_idces, return_counts=True)
    assert bin_num_vals_counts.sum() == len(probs)

    # eliminate duplicate bins, which can arise because of highly repetitive values covering multiple quantile values
    bin_arr = np.unique(bin_arr)

    if not len(bin_arr) == num_bins or \
            (min_vals_per_bin is not None and (bin_num_vals_counts < min_vals_per_bin).any()):
        # Because of some highly repetitive values, some quantiles coincide at the same values, or some bins contain
        # very few (or no) values. Attempt to recover.

        num_unique_probs = len(np.unique(probs))

        # find how many different unique values are in the current bins
        bin_idces = np.searchsorted(bin_arr, probs)
        _, bin_num_vals_counts = np.unique(bin_idces, return_counts=True)

        assert bin_num_vals_counts.sum() == len(probs)

        # Try to split/merge bins until all problems are sorted out: exactly the desired amount of bins, and all bins
        # of sufficient size (if min_vals_per_bin is not None).
        # The following loop could be handled a lot more efficiently. We should not go there too often anyway, however.
        while len(bin_arr) < num_bins or \
                (min_vals_per_bin is not None and (bin_num_vals_counts < min_vals_per_bin).any()):

            # Find unique values contained in each of the current bins
            bin_unique_vals = []
            for bin_idx in range(len(bin_arr)):
                bin_unique_vals.append(np.unique(probs[bin_idces == bin_idx]))
            bin_unique_vals_count = [len(unique_vals) for unique_vals in bin_unique_vals]
            assert len(bin_unique_vals_count) == len(bin_num_vals_counts)
            assert sum(bin_unique_vals_count) == num_unique_probs

            # Of the bins with at least two unique values, break up one of the largest ones, such that each of the new
            # bins will have at least min_vals_per_bin vals (if that is not None).
            multi_value_bin_idces = [idx for idx in range(len(bin_arr)) if bin_unique_vals_count[idx] >= 2]
            multi_value_bin_sizes = [bin_num_vals_counts[idx] for idx in multi_value_bin_idces]

            if min_vals_per_bin is not None and not any([size >= 2*min_vals_per_bin for size in multi_value_bin_sizes]):
                # It's impossible to split anything while also getting sufficiently large bins.
                raise ValueError

            # Sort by current bin size (we'll start trying to split from the largest one).
            multi_value_bin_size_and_idces_sorted = sorted(zip(multi_value_bin_sizes, multi_value_bin_idces),
                                                           key=lambda pair: pair[0], reverse=True)
            candidate_bins_for_splitting = [(bin_size, idx) for (bin_size, idx) in multi_value_bin_size_and_idces_sorted
                                            if min_vals_per_bin is None or bin_size >= 2 * min_vals_per_bin]

            found_one = False
            for bin_size, idx in candidate_bins_for_splitting:
                # We have a candidate bin to be split up.
                # How many unique values are contained in this bin?
                vals = probs[bin_idces == idx]
                local_bin_unique_vals, local_bin_unique_vals_counts = np.unique(vals, return_counts=True)
                assert local_bin_unique_vals_counts.sum() == bin_size

                # split this bin into two roughly equally sized chunks
                counts_cumsum = local_bin_unique_vals_counts.cumsum()
                optimal_splitting_idx = np.argmin(np.abs(counts_cumsum - bin_size / 2))

                if min_vals_per_bin is not None:
                    # Will the two split parts both have size > min_vals_per_bin?
                    if counts_cumsum[optimal_splitting_idx] < min_vals_per_bin or \
                            (bin_size - counts_cumsum[optimal_splitting_idx]) < min_vals_per_bin:
                        # one of the two new splits would be smaller than desired; try the next candidate for splitting
                        continue

                # Split the bin, update bin array
                bin_arr = np.delete(bin_arr, idx)
                bin_arr = np.insert(bin_arr, idx, [local_bin_unique_vals[optimal_splitting_idx],
                                                   local_bin_unique_vals[-1]])
                found_one = True
                break

            if not found_one:
                raise ValueError

            # Housekeeping
            # find how many unique values are in the remaining bins
            bin_idces = np.searchsorted(bin_arr, probs)
            _, bin_num_vals_counts = np.unique(bin_idces, return_counts=True)

            if len(bin_arr) > num_bins:
                # We came here because one bin was too small (but non-empty), not because of empty bins.
                # Now we have one bin too much. Merge two neighboring bins to eliminate one of the too-small ones.
                bin_idx_to_be_merged = np.argmin(bin_num_vals_counts)
                if 0 < bin_idx_to_be_merged < len(bin_arr) - 1:
                    # there are two neighboring bins, merge with the smaller of the two
                    if bin_num_vals_counts[bin_idx_to_be_merged - 1] < bin_num_vals_counts[bin_idx_to_be_merged + 1]:
                        bin_idx_to_be_merged = bin_idx_to_be_merged - 1
                elif bin_idx_to_be_merged == len(bin_arr) - 1:
                    bin_idx_to_be_merged = len(bin_arr) - 2
                bin_arr = np.delete(bin_arr, bin_idx_to_be_merged)

                # Re-do previous housekeeping
                bin_idces = np.searchsorted(bin_arr, probs)
                _, bin_num_vals_counts = np.unique(bin_idces, return_counts=True)

            assert bin_num_vals_counts.sum() == len(probs)
            assert len(bin_arr) <= num_bins

    bins: Bins = [bin_arr[idx] for idx in range(len(bin_arr))]

    if return_counts:
        return bins, bin_num_vals_counts
    else:
        return bins


def get_discrete_bins(data: List[float]) -> Bins:
    # Fully copied from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT License)
    sorted_values = sorted(np.unique(data))
    bins = []
    for i in range(len(sorted_values) - 1):
        mid = (sorted_values[i] + sorted_values[i+1]) / 2.0
        bins.append(mid)
    bins.append(1.0)
    return bins


def difference_mean(data: Data) -> float:
    """Returns average pred_prob - average label."""
    # Fully copied from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT License)
    data = np.array(data)
    ave_pred_prob = np.mean(data[:, 0])
    ave_label = np.mean(data[:, 1])
    return ave_pred_prob - ave_label


def get_bin_probs(binned_data: BinnedData) -> List[float]:
    # Fully copied from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT License)
    bin_sizes = list(map(len, binned_data))
    num_data = sum(bin_sizes)
    bin_probs = list(map(lambda b: b * 1.0 / num_data, bin_sizes))
    assert(abs(sum(bin_probs) - 1.0) < eps)
    return list(bin_probs)


def bin(data: Data, bins: Bins):
    # Fully copied from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT License)
    # bin boundaries are part of the _left_ bin.
    prob_label = np.array(data)
    bin_indices = np.searchsorted(bins, prob_label[:, 0])
    bin_sort_indices = np.argsort(bin_indices)
    sorted_bins = bin_indices[bin_sort_indices]
    splits = np.searchsorted(sorted_bins, list(range(1, len(bins))))
    binned_data = np.split(prob_label[bin_sort_indices], splits)
    return binned_data


def unbiased_l2_ce(binned_data: BinnedData, abort_if_not_monotonic=False) -> float:
    # Calibration error RMSE
    # The actual computation is copied from
    # https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT License)
    # I added the abort_if_not_monotonic feature, which is used in bin sweeps. Also changed the behavior if
    # len(data) < 2 to now raise an error instead of silently returning 0.
    def bin_error(data: Data):
        if len(data) < 2:
            #return 0.0
            raise ValueError('Too few values in bin, use fewer bins or get more data.')
        biased_estimate = abs(difference_mean(data)) ** 2
        label_values = list(map(lambda x: x[1], data))
        mean_label = np.mean(label_values)
        variance = mean_label * (1.0 - mean_label) / (len(data) - 1.0)
        return biased_estimate - variance

    if abort_if_not_monotonic:
        last_incidence = 0.0
        for bin_data in binned_data:
            if np.shape(bin_data)[0] == 0:
                pass
            bin_incidence = np.mean(bin_data[:, 1])
            if bin_incidence < last_incidence:
                raise ValueError("Bin incidence are non-monotonic")
            else:
                last_incidence = bin_incidence

    bin_probs = get_bin_probs(binned_data)
    bin_errors = list(map(bin_error, binned_data))

    return max(np.dot(bin_probs, bin_errors), 0.0) ** 0.5


def get_unbiased_calibration_rmse(labels, probs, num_bins="sweep", binning_scheme=get_equal_mass_bins):
    # Partially copied from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT License)
    # Bin sweep functionality added by me.
    assert probs.shape == labels.shape
    assert len(probs.shape) == 1
    if len(probs) < 10:
        return np.nan
    data = list(zip(probs, labels))
    if num_bins == 'sweep':
        assert not binning_scheme == get_discrete_bins
        # having less than ~10 samples per bin doesn't make a lot of sense, nor does having more than 100 bins
        upper_bound = min(min(round(len(probs) / 10), len(np.unique(probs))), 100)
        curr_num_bins = min(10, upper_bound)
        lower_bound = 1
        if VERBOSE:
            print('Starting bin count sweep (in Kumar)...')
        while not lower_bound == upper_bound == curr_num_bins:
            if VERBOSE:
                print(f'nbin={curr_num_bins}')
            try:
                bins = binning_scheme(probs, num_bins=curr_num_bins, min_vals_per_bin=10)
                err = unbiased_l2_ce(bin(data, bins), abort_if_not_monotonic=True)
            except ValueError:
                upper_bound = curr_num_bins - 1
                curr_num_bins = round(lower_bound + (upper_bound - lower_bound) / 2)
            else:
                lower_bound = curr_num_bins
                curr_num_bins = min(curr_num_bins * 2, round(lower_bound + (upper_bound - lower_bound) / 2))
            curr_num_bins = min(upper_bound, max(lower_bound + 1, curr_num_bins))

        if curr_num_bins == 1:
            bins = binning_scheme(probs, num_bins=curr_num_bins)
            err = unbiased_l2_ce(bin(data, bins))
        if VERBOSE:
            print(f'Final nbin={curr_num_bins}')
        return err
    else:
        if binning_scheme == get_discrete_bins:
            assert (num_bins is None)
            bins = binning_scheme(probs)
        else:
            bins = binning_scheme(probs, num_bins=num_bins)
        return unbiased_l2_ce(bin(data, bins))


class BetaCalibratedClassifier(RegressorMixin, BaseEstimator):

    def __init__(self, base_clf, beta_params="abm"):
        self.base_clf = base_clf
        self.beta_model = BetaCalibration(parameters=beta_params)

    def fit(self, X_cal, y_cal):
        y_preds = self.base_clf.predict_proba(X_cal)
        # this is an N x 2 matrix with the probabilities of the two classes
        assert (sum(abs(y_preds.sum(axis=1) - 1) < 1e-7) == len(y_preds))
        # reduce it to just the likelihood of the "1" class, as usual
        scores = y_preds[:, 1]
        self.beta_model.fit(scores, y_cal)

    def predict_proba(self, X_test):
        y_preds = self.base_clf.predict_proba(X_test)
        # this is an N x 2 matrix with the probabilities of the two classes
        assert (sum(abs(y_preds.sum(axis=1) - 1) < 1e-7) == len(y_preds))
        # reduce it to just the likelihood of the "1" class, as usual
        scores = y_preds[:, 1]
        return self.beta_model.predict(scores)

    def __repr__(self):
        return "Beta calibrated model with \n " + self.beta_model.calibrator_.lr_.__repr__() + \
            "\n and base model \n" + self.base_clf.__repr__()
