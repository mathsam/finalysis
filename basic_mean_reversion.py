# basic mean reversion alpha
# -ret1
import matplotlib.pyplot as plt
import simulator
import numpy as np
import trans

def mean_reversion(mysim):
    return -mysim['ret1']
    
def mc_rmmid(mysim):
    return trans.cs_remove_middle(-mysim['ret1'], 10)

def mr_selectweekday(mysim):
    """
    do not trade on Thursday
    """
    alpha = -mysim['ret1']
    date_index = [idate.weekday()==3-mysim.delay for idate in mysim.dates]
    date_index = np.array(date_index, dtype=np.bool)
    alpha[:,date_index] = np.nan
    return alpha

def mr_selectmonth(mysim):
    """
    do not trade on Jan, Jun, Aug, Sep
    """
    avoid_months = [1, 8, 9]
    alpha = -mysim['ret1']
    date_index= [idate.month in avoid_months for idate in mysim.dates]
    date_index = np.array(date_index, dtype=np.bool)
    alpha[:,date_index] = np.nan
    return alpha
    

mysim = simulator.Simulator(mean_reversion,retain_alpha_sign=True,
                            universe='top500',delay=1)
pnl = mysim.eval_pnl(mean_reversion)
#pnl1 = mysim.eval_pnl(mr_selectweekday)
#pnl2 = mysim.eval_pnl(mr_selectmonth)
pnl_rmmid = mysim.eval_pnl(mc_rmmid)
##
plt.plot(mysim.dates, np.cumsum(pnl), label='basic mean reversion')
#plt.plot(mysim.dates, np.cumsum(pnl1), label='avoid Thursday')
#plt.plot(mysim.dates, np.cumsum(pnl2), label='avoid Jan/Aug/Sep')
plt.plot(mysim.dates, np.cumsum(pnl_rmmid), label='remove mid mean reversion')
plt.legend(loc='best')
plt.show()
##
pnl_dates   = mysim.add_dates_keys(pnl=pnl)
pnl_weekday = pnl_dates['pnl'].groupby(pnl_dates['weekday'])
pnl_month   = pnl_dates['pnl'].groupby(pnl_dates['month'])
pnl_year    = pnl_dates['pnl'].groupby(map(str,pnl_dates['year']))

## plot the results
## monthly
trading_days_each_month = np.array([19, 19, 23, 21, 20, 22, 21, 22, 21, 21, 21, 22])
pnl_month_ave = pnl_month.mean()*trading_days_each_month
pnl_month_std = pnl_month.std()*np.sqrt(trading_days_each_month)
ax_month = pnl_month_ave.plot(kind='bar',yerr=[pnl_month_std, pnl_month_std])
ax_month.set_title('average monthly pnl and std')
ax_month.set_ylabel('monthly pnl')
plt.show()
##
ax_weekday = pnl_weekday.mean().plot(kind='bar',yerr=[pnl_weekday.std(), pnl_weekday.std()])
ax_weekday.set_title('average daily pnl and std')
ax_weekday.set_ylabel('daily pnl')
plt.show()