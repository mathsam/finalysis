# basic mean reversion alpha
# -ret1
import matplotlib.pyplot as plt
import simulator
import numpy as np
import trans
import pandas as pd

def mc_basic(which_ret, threshold=0.1):
    def mc_inner(mysim):
        alpha = -mysim[which_ret]
        alpha[alpha>threshold]  = np.nan
        alpha[alpha<-threshold] = np.nan
        return alpha
    return mc_inner

def mc_rmmid(which_ret, percent=10, lower_threshold=0.02, upper_threshold=0.1):
    def mc_rmmid_inner(mysim):
        alpha = trans.cs_remove_middle(-mysim[which_ret], percent)
        alpha[np.abs(alpha)> upper_threshold] = np.nan
        alpha[np.abs(alpha)< lower_threshold] = np.nan
        return alpha
    return mc_rmmid_inner


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


mysim = simulator.Simulator(None,retain_alpha_sign=True,
                            universe='top2000',delay=1)
pnl1, s1, eachret1, raw_ret = mysim.eval_pnl(mc_rmmid('ret1',10, 0.01, 0.2))
#pnl2, s2, eachret2 = mysim.eval_pnl(mc_rmmid('open_ret1',10, 1))                            

#pnl1, s1, eachret1 = mysim.eval_pnl(mc_rmmid('ret1',10,10))
#pnl2, s2, eachret2 = mysim.eval_pnl(mc_rmmid('open_ret1',10,10))
#pnl2 = mysim.eval_pnl(mr_selectmonth)
#pnl_int, s2 = mysim.eval_pnl(mc_basic('open_ret1'), 'open2open')
##
plt.plot(mysim.dates, np.cumsum(pnl1), label='close to close ret1')
#plt.plot(mysim.dates, np.cumsum(pnl1), label='avoid Thursday')
#plt.plot(mysim.dates, np.cumsum(pnl2), label='avoid Jan/Aug/Sep')
plt.plot(mysim.dates, np.cumsum(pnl2), label='open to open ret1')
plt.plot(mysim.dates, np.cumsum(pnl1+pnl2)/2., label='average')
plt.title('cs_remove_middle(-ret1,0.8) alpha')
plt.legend(loc='best')
plt.show()