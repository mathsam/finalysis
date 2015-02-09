from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import stats_util
import trans

#volume_mean = np.nanmean(mysim('volume'), 1)
#volume_anomaly = mysim('volume')/volume_mean[:, np.newaxis]

#delayed_days = 8;
#time_range   = slice(None, None)
#x_y = DataFrame({'x':mysim('ret1')[time_range,:-delayed_days-2].flatten(),
#                 'y':mysim('vwap_ret1')[time_range,delayed_days+2:].flatten()})

x_y = DataFrame({'x':trans.ts_delay(mysim('open')-mysim('close'), 8).flatten(),
                 'y':raw_ret.flatten()})

quantile_rank, binmid = stats_util.quantile(x_y.x, 20)
x_y['quantile'] = quantile_rank
x_y['bin']      = binmid
x_y_bybin      = x_y.groupby('bin')
y_bybin_mean = x_y_bybin.y.mean()

##plotting
fig = plt.figure()
ax  = fig.add_subplot(111)
y_bybin_mean.sort(inplace=0)
plt.plot(y_bybin_mean.index.values, y_bybin_mean.values, '--o')
ax.set_xlabel('open-close before trading day')
ax.set_ylabel('HF momentum daily PnL')
#ax.set_xscale('log')
plt.show()


##
