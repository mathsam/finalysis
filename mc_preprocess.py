import numpy as np

def series_to_nfeatures(data2d, num_features, additional_feature=None):
    """from time series `data2d` select past `num_features` times to be used as 
    features. For example, time series [0, 1, 3, 5] and num_features = 2 becomes
       [[nan, 0],
        [0, 1],
        [1, 3],
        [3, 5]]
    treat all the stocks as the same
    """
    from trans import ts_delay
    num_stocks = data2d.shape[0]
    num_times  = data2d.shape[1]
    if additional_feature is None:
        feature_vec = np.zeros((num_times*num_stocks, num_features))
    else:
        feature_vec = np.zeros((num_times*num_stocks, num_features+1))
    for i in range(0, num_features):
        feature_vec[:,i] = ts_delay(data2d, i).flatten()
    if additional_feature is not None:
        feature_vec[:,num_features] = np.tile(additional_feature, (1, num_stocks))
    return feature_vec