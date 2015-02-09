import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pandas as pd
import trans
from pandas import DataFrame
np.seterr(invalid='ignore')

database_dir = '/media/junyic/Work/Trexquant/database/stocks/'
stock_universes = ['top100', 'top250', 'top500', 'top1000',
                   'top2000','top2500','top3000']
data_valid_range = {'ret1':[-1., 1.],'vwap_ret1':[-.5, .5],'ret5':[-1., 1.]}

class Simulator(object):
    """
    evaluate PnL, ir and other statistics for a zero investment stategy
    investing `capital` long and `capital` short positions, and use `capital` as
    the unit in PnL
    """
    def __init__(self, alpha=None, retain_alpha_sign=True, universe='top2000',
                 delay=1, capital=1e6):
        if universe in stock_universes:
            self.universe = universe
            uni_file = database_dir + universe + '.adj_ammend.mat'
            uni_var  = scipy.io.loadmat(uni_file)[universe].astype(float)
            uni_var[uni_var==0.] = np.nan
            self._uni_mask = uni_var
        else:
            raise KeyError('stock universe does not exist')
        self.alpha = alpha
        self.retain_alpha_sign = retain_alpha_sign
        if not isinstance(delay, int) or delay < 0:
            raise ValueError('delay must be int and >=0')
        self.delay = delay
        self.capital = capital
        self._market_data = {}
        
        dates = Simulator._read_var('dates')
        dates = [time.strptime(str(idate),"[%Y%m%d]") for idate in dates]
        dates = [datetime.datetime(*idate[0:6]) for idate in dates]
        self.dates = dates
        
        open_p = Simulator._read_var('open',self._uni_mask)
        open_ret1 = open_p/trans.ts_delay(open_p,1) - 1.0
        self._market_data['open_ret1']  = open_ret1
        
        close_p = Simulator._read_var('close',self._uni_mask)
        close_ret1 = close_p/trans.ts_delay(close_p,1) - 1.0
        self._market_data['close_ret1'] = close_ret1
        
        vwap = Simulator._read_var('vwap', self._uni_mask)
        vwap_ret1 = vwap/trans.ts_delay(vwap,1) - 1.0
        Simulator._clean_data('vwap_ret1', vwap_ret1)
        forward_vwap_ret1 = trans.ts_delay(vwap_ret1, -1-self.delay)
        self._market_data['vwap_ret1'] = vwap_ret1
        self._market_data['forward_vwap_ret1'] = forward_vwap_ret1
        return
        
    @staticmethod
    def _read_var(varname, mask=None):
        var = scipy.io.loadmat(database_dir+varname+'.adj_ammend.mat')[varname]
        if mask is not None:
            var *= mask
        Simulator._clean_data(varname, var)
        return var

    @staticmethod
    def _clean_data(varname, var):
        """
        clean the data if valid ranges are specified
        """
        if varname in data_valid_range:
            var[var<data_valid_range[varname][0]] = np.nan
            var[var>data_valid_range[varname][1]] = np.nan
        return None
        
    def __call__(self, varname):
        """
        Usage:
            mysim('ret1') #returns a reference of numpy array for the variable
        Note:
            as it returns a referece, use with care
        """
        if varname in self._market_data:
            return self._market_data[varname].copy()
        var = Simulator._read_var(varname, self._uni_mask)
        self._market_data[varname] = var
        return var
        
    def __getitem__(self, varname):
        """
        Usage:
            mysim['ret'] # returns a copy of numpy array for the variable
        """
        return self(varname).copy()
    
    def eval_pnl(self,alpha=None,allow_frac=False):
        if alpha:
            self.alpha = alpha
        signal  = self.alpha(self).astype(float)
        raw_ret = self('forward_vwap_ret1')*signal
        # normalized the signal into zero investment strategy
        if not self.retain_alpha_sign:
            signal_mean = np.nanmean(signal, axis=0)[np.newaxis,:]
            signal -= signal_mean

        p_index  = (signal>0.).astype(np.int8)
        n_index  = (signal<0.).astype(np.int8)
        p_signal = signal*p_index
        n_signal = signal*n_index
        p_norm   = np.nansum(p_signal, axis=0)[np.newaxis,:]
        n_norm   = -np.nansum(n_signal, axis=0)[np.newaxis,:]
        signal   = p_signal/p_norm + n_signal/n_norm
        
        if not allow_frac:
            signal = self.capital*signal/self('vwap')
            signal = np.trunc(signal)*self('vwap')/self.capital

        each_stock_return = self('forward_vwap_ret1')*signal

        pnl = np.nansum(each_stock_return, axis=0)
        pnl[np.isnan(pnl)] = 0.
        self.pnl = pnl
        return pnl, signal, each_stock_return, raw_ret
        
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