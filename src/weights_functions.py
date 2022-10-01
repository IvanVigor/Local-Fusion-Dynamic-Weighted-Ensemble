from src.metrics import *

def w_inverse_LMAE(actual, predicted):
    """
    Inverse Local MAE
    """
    return 1/mean_absolute_error(actual, predicted)

def w_inverse_log_LMAE(residuals):
    """
    Inverse Log Local MAE
    """
    np.log(max(abs(residuals))/mean_absolute_error(residuals))