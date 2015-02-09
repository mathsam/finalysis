from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import stats_util
import trans

#volume_mean = np.nanmean(mysim('volume'), 1)
#volume_anomaly = mysim('volume')/volume_mean[:, np.newaxis]

delayed_days = 10;
time_range   = slice(None, None)
#x_y = DataFrame({'x':mysim('volume')[:,:-delayed_days].flatten(),
#                 'y':eachret1[:,delayed_days:].flatten()})
#x_y = DataFrame({'x':mysim('ret1').flatten(),
#                 'y':raw_ret.flatten()})

#ret1_delayed = trans.ts_delay(mysim('ret1'), delayed_days)
#ret1_delayed[np.abs(ret1_delayed)<0.04] = np.nan
#ret1         = mysim['ret1']
#ret1[np.abs(ret1)<0.04] = np.nan
x_y = DataFrame({'x':np.nanstd(mysim('ret1'),0),
                 'y':pnl1})

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
ax.set_xlabel('volume %d days ago' %delayed_days)
ax.set_ylabel('momenHF_PnL (8 day delay)')
#ax.set_xscale('log')
plt.show()


##
