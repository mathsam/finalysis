import pandas as pd
import matplotlib.pyplot as plt
import stats_util
import os

future_dir = '/media/junyic/Work/Trexquant/database/future/'
fileset    = os.listdir(future_dir)
savedir    = '/media/junyic/Work/Trexquant/project5/future_corr/'

pnl_pd = pd.DataFrame(pnl1, columns=['pnl'], index = mysim.dates)
pnl_pd.index.name = 'date'

for filename in fileset:
    fig = plot_xcorr(future_dir + filename, pnl_pd)
    fig.savefig(savedir + filename + '.png')
    
## generate pnl dateframe
def plot_xcorr(file_fullpath, pnl_pd):
    future_var = pd.read_table(file_fullpath,
                            skiprows=[0,1,2,3,4,5,6],
                            header=None,
                            names=['date','value','null'],
                            index_col = 'date',
                            parse_dates=True,
                            dayfirst=True)
    future_var.drop('null',axis=1,inplace=True)
    future_var_ret = future_var.pct_change(periods=1)
    
    pnl_vs_future = pd.merge(pnl_pd, future_var_ret, left_index=True, right_index=True, how='outer')
    
    # pnl and future_ret that shares the same dates
    pnl_common     = pnl_vs_future['pnl'].values
    future_common  = pnl_vs_future['value'].values
    lags1, r1, pr1 = stats_util.aggre_xcorr(pnl_common, future_common, 20, 1)
    lags5, r5, pr5 = stats_util.aggre_xcorr(pnl_common, future_common, 20, 5)
    lags10,r10,pr10= stats_util.aggre_xcorr(pnl_common, future_common, 20, 10)
    
    significant_level = 0.05
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(lags1,  r1, '--ro', label='daily')
    ax.plot(lags1[pr1<significant_level], r1[pr1<significant_level], 'rs', ms=10)
    ax.plot(lags5,  r5, '--go', label='5 day aggre')
    ax.plot(lags5[pr5<significant_level], r5[pr5<significant_level], 'gs', ms=10)
    ax.plot(lags10, r10,'--bo', label='10 day aggre')
    ax.plot(lags10[pr10<significant_level],r10[pr10<significant_level], 'bs', ms=10)
    ax.legend(loc='best')
    ax.set_xlabel('lag')
    ax.set_ylabel('correlation')
    plt.close()
    return fig