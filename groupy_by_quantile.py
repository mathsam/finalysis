from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import stats_util
import trans

#volume_mean = np.nanmean(mysim('volume'), 1)
#volume_anomaly = mysim('volume')/volume_mean[:, np.newaxis]

#delayed_days = 120;
#time_range   = slice(None, None)
#x_y = DataFrame({'x':mysim('volume')[:,:-delayed_days].flatten(),
#                 'y':eachret1[:,delayed_days:].flatten()})
#ret1_shock = mysim['ret1']
#ret1_shock[np.abs(ret1_shock)<0.05] = np.nan
#ret_smoothed = trans.ts_mean(mysim('ret1'), 120)
#x_y = DataFrame({'x':trans.ts_delay(ret_smoothed, delayed_days).flatten(),
#                 'y':ret_smoothed.flatten()})
                 
x_y = DataFrame({'x': smoothed_ret.flatten(),
                 'y': raw_ret.flatten()})   

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
ax.set_xlabel('ts_mean(ret1, %s)' %delayed_days)
ax.set_ylabel('ave ret')
#ax.set_xscale('log')
plt.show()


##
