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
        return trans.ts_delay(np.sign(mysim(varname)), delayed_days)
    return momen_HF_inner
    
mysim = simulator.Simulator(None,retain_alpha_sign=True,
                            universe='top2000',delay=1)
#pnl1, s1, eachret1, raw_ret = mysim.eval_pnl(momen_longterm([-250, -7], 90))
pnl1, s1, eachret1, raw_ret = mysim.eval_pnl(momen_HF('ret1', 8))