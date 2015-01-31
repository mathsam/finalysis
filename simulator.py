import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pandas as pd
from pandas import DataFrame
np.seterr(invalid='ignore')

database_dir = '/media/junyic/Work/Trexquant/database/stocks/'
stock_universes = ['top100', 'top250', 'top500', 'top1000',
                   'top2000','top2500','top3000']
data_valid_range = {'ret1':[-1., 1.]}

class Simulator(object):
    """
    evaluate PnL, ir and other statistics for a zero investment stategy
    investing 1$ long and 1$ short positions
    """
    def __init__(self, alpha=None, retain_alpha_sign=True, universe='top2000',
                 delay=1):
        if universe in stock_universes:
            self.universe = universe
            uni_file = database_dir + universe + '.adj_ammend.mat'
            uni_var  = scipy.io.loadmat(uni_file)[universe].astype(np.int8)
            self._uni_index = uni_var
        else:
            raise KeyError('stock universe does not exist')
        self.alpha = alpha
        self.retain_alpha_sign = retain_alpha_sign
        if not isinstance(delay, int) or delay < 0:
            raise ValueError('delay must be int and >=0')
        self.delay = delay
        self._market_data = {}
        
        dates = scipy.io.loadmat(database_dir+'dates.adj_ammend.mat')['dates']
        dates = [time.strptime(str(idate),"[%Y%m%d]") for idate in dates]
        dates = [datetime.datetime(*idate[0:6]) for idate in dates]
        self.dates = dates
        return
        
    def __getitem__(self, varname):
        """
        Usage:
            mysim['ret1'] #returns a numpy array for that variable
        """
        if varname in self._market_data:
            return self._market_data[varname].copy()
        varfile = database_dir + varname + '.adj_ammend.mat'
        var = scipy.io.loadmat(varfile)[varname]*self._uni_index
        
        #clean the data if valid ranges are specified
        if varname in data_valid_range:
            var[var<data_valid_range[varname][0]] = np.nan
            var[var>data_valid_range[varname][1]] = np.nan

        self._market_data[varname] = var
        return var.copy()
    
    def eval_pnl(self,alpha=None, delay=1):
        if alpha:
            self.alpha = alpha
        signal = self.alpha(self)
        # normalized the signal into zero investment strategy
        if not self.retain_alpha_sign:
            signal_mean = np.nanmean(signal, axis=0)[np.newaxis,:]
            signal -= signal_mean

        p_index  = (signal>0.).astype(np.int8)
        n_index  = (signal<0.).astype(np.int8)
        p_signal = signal*p_index
        n_signal = signal*n_index
        p_norm    = np.nansum(p_signal, axis=0)[np.newaxis,:]
        n_norm    = -np.nansum(n_signal, axis=0)[np.newaxis,:]
        signal   = p_signal/p_norm + n_signal/n_norm
        
        if self.delay > 0:
            delayed_signal = np.empty_like(signal)
            delayed_signal[:,delay:]  = signal[:,0:-delay]
            delayed_signal[:,0:delay] = np.nan
        else:
            delayed_signal = signal

        each_stock_return = self['ret1']*delayed_signal
        pnl = np.nansum(each_stock_return, axis=0)
        pnl[np.isnan(pnl)] = 0.
        self.pnl = pnl
        return pnl
        
    def visualize(self, if_show=True):
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(self.dates, np.cumsum(self.pnl))
        ax.set_xlabel('Time')
        ax.set_ylabel('PnL')
        if if_show:
            plt.show()
        return fig
        
    def add_dates_keys(self, **kwargs):
        """
        Usage:
            mysim.add_dates_keys(pnl=pnl,volume=vol) # use default dates
            mysim.add_dates_keys(dates=mydates, pnl=pnl) # use mydates
        """
        if 'dates' not in kwargs or kwargs['dates'] is None:
            dates = self.dates
        else:
            dates = kwargs['dates']
        kwargs.pop('dates',None)
        for var in kwargs:
            if len(kwargs[var]) != len(dates):
                raise IndexError('length of user_data does not match dates')

        dayofweek = [idate.weekday() for idate in dates]
        months    = [idate.month for idate in dates]
        years     = [idate.year  for idate in dates]
        days      = [idate.day   for idate in dates]
        kwargs['weekday'] = dayofweek
        kwargs['month']  = months
        kwargs['year']   = years
        kwargs['day']    = days
        keyed_data = DataFrame(kwargs)
        return keyed_data