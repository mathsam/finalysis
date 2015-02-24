ret1 = mysim['ret1']
retmean = np.nanmean(mysim('ret1'), 0)
retstd  = np.nanstd(mysim('ret1'), 0)
ret1_shock = np.zeros_like(ret1)

ret1_shock[(ret1 - retmean[np.newaxis,:]) > 2*retstd[np.newaxis,:]] = 1
ret1_shock[(ret1 - retmean[np.newaxis,:]) <-2*retstd[np.newaxis,:]] = -1
## time_to_lastshock
import stats_util
from pandas import DataFrame
time2pshock, time2nshock = stats_util.time_to_lastshock(ret1_shock)
##
ret_vs_shock = DataFrame({'ret': mysim['ret1'].flatten(),
                           'time2nshock': time2nshock.flatten()})
ave_ret_byshock = ret_vs_shock.groupby('time2nshock').ret.mean()
ave_ret_byshock.plot()
plt.ylabel('ave return')
plt.show()

## plot the number of shock vs. date
num_shocks  = np.sum(np.abs(ret1_shock), 0)
ret1_pshock = np.zeros_like(ret1)
ret1_pshock[(ret1 - retmean[np.newaxis,:]) > 2*retstd[np.newaxis,:]] = 1
ret1_nshock = np.zeros_like(ret1)
ret1_nshock[(ret1 - retmean[np.newaxis,:]) <-2*retstd[np.newaxis,:]] = -1
num_pshocks = np.sum(ret1_pshock, 0)
num_nshocks =np.sum(ret1_nshock, 0)
plt.xlabel('date')
plt.ylabel('num of shocks')
plt.plot(mysim.dates, num_shocks, label='total shocks')
plt.plot(mysim.dates, num_pshocks,label='positive shocks')
plt.plot(mysim.dates, num_nshocks,label='negative shocks')
plt.legend(loc='best')
plt.show()