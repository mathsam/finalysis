import numpy as np
from scipy.stats import pearsonr

def xcorr(x, y, maxlags=20):
    """
    lagged correlation
    """
    if not (isinstance(x, np.ndarray) and 
            isinstance(y, np.ndarray)):
        raise TypeError('Inputs must be numpy.ndarray')
    
    if len(x)!=len(y):
        raise ValueError('Input variables of different lengths.')
            
    valid_index = np.logical_and(~np.isnan(x), ~np.isnan(y))
    valid_x = x[valid_index]
    valid_y = y[valid_index]

    if maxlags >= len(valid_x):
        raise ValueError('Max lag >= valid length of input signals')
        
    lags = np.arange(-maxlags, maxlags+1, 1)
    r    = np.empty_like(lags, dtype=float)
    pr   = np.empty_like(lags, dtype=float)
    for i, ilag in enumerate(lags):
        if ilag < 0:
            r[i], pr[i] = pearsonr(x[:ilag],y[-ilag:])
        elif ilag ==0:
            r[i], pr[i] = pearsonr(x,y)
        elif ilag>0:
            r[i], pr[i] = pearsonr(x[ilag:],y[:-ilag])
    return lags, r, pr

def aggre_xcorr(x, y, maxlags=20, aggre_width=5):
    if not (isinstance(x, np.ndarray) and 
            isinstance(y, np.ndarray)):
        raise TypeError('Inputs must be numpy.ndarray')
    
    if len(x)!=len(y):
        raise ValueError('Input variables of different lengths.')
            
    valid_index = np.logical_and(~np.isnan(x), ~np.isnan(y))
    valid_x = x[valid_index]
    valid_y = y[valid_index]
    valid_x = smooth(valid_x, window_len=aggre_width, window='flat')
    valid_y = smooth(valid_y, window_len=aggre_width, window='flat')

    if maxlags*aggre_width >= len(valid_x):
        raise ValueError('maxlags*aggre_width >= valid length of input signals')

    lags = np.arange(-maxlags, maxlags+1, 1)
    r    = np.empty_like(lags, dtype=float)
    pr   = np.empty_like(lags, dtype=float)
    for i, ilag in enumerate(lags):
        if ilag < 0:
            r[i], pr[i] = pearsonr(valid_x[:ilag:aggre_width],valid_y[-ilag::aggre_width])
        elif ilag ==0:
            r[i], pr[i] = pearsonr(valid_x[::aggre_width],valid_y[::aggre_width])
        elif ilag>0:
            r[i], pr[i] = pearsonr(valid_x[ilag::aggre_width],valid_y[:-ilag:aggre_width])
    return lags, r, pr

def smooth(x, window_len=10, window='flat'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]