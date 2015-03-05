"""
find parameters that optimize the PnL
"""
import simulator
mysim = simulator.Simulator(None,retain_alpha_sign=True,
                            universe='top2000',delay=1)
import trans              
import numpy as np          

def linear_alpha(ks):
    def linear_alpha_inner(mysim):
        alpha = mysim['ret1']
        for i in range(0, 14):
            alpha += ks[i]*trans.ts_delay(mysim('ret1'), i)
        alpha += ks[14]*np.log(mysim('volume'))
        return alpha
    return linear_alpha_inner

def linear_strategy_pnl(k):
    pnl1, s1, eachret1, raw_ret = mysim.eval_pnl(linear_alpha(k))
    return -np.mean(pnl1)/np.std(pnl1)
    
## mixed mean reversion and long term momentum
ret1         = mysim['ret1']
ret_longterm = trans.ts_delay(
    1.0 - trans.ts_delay(mysim('close'), 100)/mysim('close'),
    14)
log_volume = np.log(mysim['volume'])
def mr_momem_alpha(ks):
    def mr_momem_alpha_inner(mysim):
        alpha = -ret1
        alpha+= ks[0]*ret_longterm
        alpha+= ks[1]*log_volume
        return alpha
    return mr_momem_alpha_inner
    
def mr_momen_pnl(ks):
    pnl1, s1, eachret1, raw_ret = mysim.eval_pnl(mr_momem_alpha(ks))
    return -np.mean(pnl1)/np.std(pnl1)
    
## minimization
from scipy.optimize import minimize

res = minimize(linear_strategy_pnl, [-1.0]*14 + [0.01], method='Nelder-Mead',
               options={'xtol': 1e-8, 'disp': True})
print res