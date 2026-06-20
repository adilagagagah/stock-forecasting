import numpy as np
from scipy.stats import linregress


def direction_label(open_t, future_close):  
    x = np.arange(1, 6)
    slope, intercept, r_value, p_value, std_err = linregress(
        x,
        future_close
    )
    # slope = slope / open_t
    return slope

def return_label(open_t, future_high):
    return_label = (future_high.max() - open_t) / open_t
    return return_label

def risk_label(open_t, future_low):
    risk_label = (future_low.min() - open_t) / open_t
    return risk_label

def create_labels(open_t, future_close, future_high, future_low):
    return {
        "trend_slope": direction_label(open_t, future_close),
        "return": return_label(open_t, future_high),
        "risk": risk_label(open_t, future_low),
        "days_to_max": np.argmax(future_high) + 1,
        "days_to_min": np.argmin(future_low) + 1,
    }