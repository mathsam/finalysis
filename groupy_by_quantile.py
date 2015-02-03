from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt

x_y = DataFrame({'x':np.nanstd(mysim('ret1'), 1),
                 'y':np.nanmean(eachret1, 1)})

quantile_rank, binmid = quantile(x_y.x, 10)
x_y['quantile'] = quantile_rank
x_y['bin']      = binmid
x_y_bybin      = x_y.groupby('bin')
y_bybin_mean = x_y_bybin.y.mean()

##ploting
fig = plt.figure()
ax  = fig.add_subplot(111)
y_bybin_mean.sort(inplace=0)
plt.plot(y_bybin_mean.index.values, y_bybin_mean.values, '--o')
ax.set_xlabel('market cross sectional volatility')
ax.set_ylabel('mean reversion average ret')
plt.show()


##
def quantile(time_series, numbins=10):
    from pandas import Series
    import pandas as pd
    import numpy as np
    q, bins = pd.qcut(time_series, numbins, retbins=True)
    which_bin = lambda x: np.nan if np.isnan(x) else np.maximum(1, np.where(x <= bins)[0][0])
    bin_left  = lambda x: np.nan if np.isnan(x) else bins[which_bin(x)-1]
    quantile_series = time_series.map(which_bin)
    binmid_series   = time_series.map(bin_left)
    return quantile_series, binmid_series