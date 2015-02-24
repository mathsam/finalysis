# study of Naive Bayes can be used to predict mean reversion behavior (daily 
# return)

ret1 = mysim['ret1']

train_days = 1500
delayed_days = 2
train_ret1  = ret1[:, 0:train_days].flatten()
train_label = np.sign(ret1[:, delayed_days:train_days+delayed_days].flatten())
train_label[train_label==0] = np.nan
cal_nan_index = lambda x, y: np.logical_or(np.isnan(x), np.isnan(y))
nan_index = cal_nan_index(train_ret1, train_label)
train_ret1  = train_ret1[~nan_index]
train_label = train_label[~nan_index]

test_ret1  = ret1[:, train_days:-delayed_days].flatten()
test_label = np.sign(ret1[:, train_days+delayed_days:]).flatten()
test_label[test_label==0] = np.nan
nan_index = cal_nan_index(test_ret1, test_label)
test_ret1  = test_ret1[~nan_index]
test_label = test_label[~nan_index]

##
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(train_ret1[:,np.newaxis], train_label)

pred_ret = gnb.predict(test_ret1[:,np.newaxis])

## metrics for prediction quality
import sklearn.metrics
report = sklearn.metrics.classification_report(test_label,
                                               pred_ret)
print report                                               