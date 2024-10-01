import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False, norm_test: bool = False) -> None:

        self.norm_test = norm_test
        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """

        norm_str = ""
        if (self.is_normalize):
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"


    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        dict_func: function reference
        """

        dist_func = None
        if self.metric == "euclidean":
            if self.is_normalize:
                return norm_ED_distance
            else:
                return ED_distance
            
        if self.metric == "dtw":
            return DTW_distance
        return dist_func


    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set
        
        Returns
        -------
        matrix_values: distance matrix
        """
        
        if (self.distance_metric != "normalized euclidean distance" and self.norm_test):
          for i in range(0, len(input_data)):
            input_data[i] = z_normalize(input_data[i])
        
        matrix_shape = (input_data.shape[0], input_data.shape[0])
        matrix_values = np.zeros(shape=matrix_shape)
        
        for i in range(0, input_data.shape[0]):
          for j in range(0, input_data.shape[0]):
            if j < i:
              matrix_values[i, j] = matrix_values[j, i]
              continue
            if i == j:
              matrix_values[i, j] = 0
              continue
            dict_func = self._choose_distance()
            matrix_values[i, j] = dict_func(input_data[i], input_data[j])

        return matrix_values
