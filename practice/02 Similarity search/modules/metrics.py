import numpy as np
import math
import numpy


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    ed_dist = 0
    if len(ts1) != len(ts2):
      raise ValueError("Временные ряды должны иметь одинаковую длину.")
    squared_differences = np.square(ts1 - ts2)
    sum_of_squared_differences = np.sum(squared_differences)
    ed_dist = np.sqrt(sum_of_squared_differences)
    return ed_dist


def simple_average_ts(ts, deviation_flag=False):
  sum = 0
  for val in ts:
    if deviation_flag:
      sum += val*val
    else:
      sum += val
  return sum/len(ts)


def standard_deviation_ts(ts):
  return math.sqrt(simple_average_ts(ts, True) - math.pow(simple_average_ts(ts, False), 2))


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0
    
    if len(ts1) != len(ts2):
        return
    n = len(ts1)
    norm_ed_dist = math.sqrt(abs(2*n*(1- (numpy.dot(ts1, ts2) - n*simple_average_ts(ts1)*simple_average_ts(ts2)) / (n*standard_deviation_ts(ts1)*standard_deviation_ts(ts2)) )))

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    dtw_dist = 0

    if len(ts1) != len(ts2):
      raise ValueError("Временные ряды должны иметь одинаковую длину.")

    n = len(ts1)
    m = len(ts2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[:, 0] = float('inf')
    dtw_matrix[0, :] = float('inf')
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
      for j in range(max(1, i - r), min(m, i + r) + 1):
        cost = (ts1[i - 1] - ts2[j - 1])*(ts1[i - 1] - ts2[j - 1])
        dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    dtw_dist = dtw_matrix[-1, -1]
    return dtw_dist
