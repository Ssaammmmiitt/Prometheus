import numpy as np

def fss(y_true, y_score, threshold, window_size):
    from scipy.ndimage import uniform_filter
    
    y = y_true >= threshold
    s = y_score >= threshold
    
    # Calculate fractional coverage
    y_frac = uniform_filter(y.astype(float), size=window_size, mode='constant', cval=0.0)
    s_frac = uniform_filter(s.astype(float), size=window_size, mode='constant', cval=0.0)
    
    mse = np.nanmean((y_frac - s_frac)**2)
    mse_ref = np.nanmean(y_frac**2 + s_frac**2)
    
    if mse_ref == 0:
        return np.nan
        
    return 1.0 - (mse / mse_ref)

def rev(y_true, y_score, c_ratio, threshold=None):
    # Relative Economic Value
    # c_ratio is cost/loss ratio
    if threshold is None:
        threshold = c_ratio
        
    hits = np.sum((y_score >= threshold) & (y_true > 0))
    false_alarms = np.sum((y_score >= threshold) & (y_true == 0))
    misses = np.sum((y_score < threshold) & (y_true > 0))
    correct_negatives = np.sum((y_score < threshold) & (y_true == 0))
    
    N = hits + false_alarms + misses + correct_negatives
    if N == 0:
        return np.nan
        
    base_rate = np.sum(y_true > 0) / N
    
    # Expected expense of climatology strategy
    # Either always act (cost = c/l) or never act (loss = base_rate)
    expense_clim = min(c_ratio, base_rate)
    
    # Expected expense of perfect forecast
    expense_perf = base_rate * c_ratio
    
    # Expected expense of our forecast
    expense_fcst = (hits + false_alarms) / N * c_ratio + (misses / N)
    
    if expense_clim - expense_perf == 0:
        return np.nan
        
    return (expense_clim - expense_fcst) / (expense_clim - expense_perf)
    
