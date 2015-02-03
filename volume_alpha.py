import trans
import matplotlib.pyplot as plt
import simulator
import numpy as np

def volume_alpha(percent=20, lookback=10):
    def va_inner(mysim):
        return trans.ts_identify_shock(mysim('volume'),percent, lookback)
    return va_inner


mysim = simulator.Simulator(None,retain_alpha_sign=True,
                            universe='top2000',delay=1)
pnl1, s1, eachret1 = mysim.eval_pnl(volume_alpha(20, 10))