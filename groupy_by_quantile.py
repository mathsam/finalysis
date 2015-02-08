from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import stats_util

#volume_mean = np.nanmean(mysim('volume'), 1)
#volume_anomaly = mysim('volume')/volume_mean[:, np.newaxis]

x_y = DataFrame({'x':mysim('ret1')[0:2000,:-2].flatten(),
                 'y':mysim('ret1')[0:2000,2:].flatten()})

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
ax.set_xlabel('today return')
ax.set_ylabel('day after tomorrow return')
#ax.set_xscale('log')
plt.show()


##
