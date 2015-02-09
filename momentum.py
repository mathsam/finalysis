# momentum strategy
import matplotlib.pyplot as plt
import simulator
import numpy as np
import trans
import pandas as pd

def momen_longterm(lookback_period = [-250, -7], hold_length = 90):
    def momen_longterm_inner(mysim):
        ave_past_ret = trans.ts_mean(mysim('ret1'),
                                     lookback_period[1]-lookback_period[0]+1)
        ave_past_ret = trans.ts_delay(ave_past_ret, -lookback_period[1])
        alpha = trans.ts_make_squarewave(ave_past_ret, hold_length)
        return alpha
    return momen_longterm_inner

def momen_HF(varname, delayed_days=8):
    def momen_HF_inner(mysim):
        return trans.ts_delay(mysim(varname), delayed_days)
    return momen_HF_inner
    
mysim = simulator.Simulator(None,retain_alpha_sign=True,
                            universe='top2000',delay=1)
#pnl1, s1, eachret1, raw_ret = mysim.eval_pnl(momen_longterm([-250, -7], 90))
pnl1, s1, eachret1, raw_ret = mysim.eval_pnl(momen_HF('ret1', 8))

## how ret1 in the future correlates with ret1 at today
import stats_util
import scipy.stats as stats
import matplotlib.pyplot as plt

look_forward_range = range(1, 25)
slopes             = np.zeros(len(look_forward_range))
r_values           = np.zeros(len(look_forward_range))
rank_vs_ret        = []

for i, look_forward in enumerate(look_forward_range):
    todays, futures = stats_util.sort_by(mysim('ret1')[:,:-look_forward].flatten(),
                                 mysim('ret1')[:,look_forward:].flatten(), 17)
    slope, intercept, r_value, p_value, std_err = stats.linregress(todays[1:-1],
                                                                  futures[1:-1])
    rank_vs_ret.append((todays, futures))                                                                  
    slopes[i]   = slope
    r_values[i] = r_value                                                                 
    
## plot rank vs mean return
fig = plt.figure()
ax  = fig.add_subplot(111)
for i in [1, 4, 9, 13, 15, 17]:
    itoday, ifuture = rank_vs_ret[i]
    ax.plot(itoday, ifuture, '--o', label='%s days later' %(i+1))
    
ax.legend(loc='best')
ax.set_xlabel('ret1 today')
ax.set_ylabel('ret1 in the future')
plt.show()
## plot slopes and r values
fig, ax1 = plt.subplots()
ax2      = ax1.twinx()
ax1.plot(look_forward_range, slopes,  '--rs', label='slope')
ax2.plot(look_forward_range, r_values,'--go', label='r_value')
ax1.set_xlabel('future days')
ax1.set_ylabel('slope',color='r')
ax2.set_ylabel('r_value',color='g')
plt.show()