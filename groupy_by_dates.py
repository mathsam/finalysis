## analyse pnl each weekday and each month
pnl_dates   = mysim.add_dates_keys(pnl=(pnl1+pnl2)/2.)
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