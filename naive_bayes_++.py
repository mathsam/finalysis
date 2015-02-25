# study of Naive Bayes can be used to predict mean reversion behavior (daily 
# return)
import trans

ret1 = mysim['ret1']
#ret1 -= np.nanmean(ret1, 0)


train_days = 1500
num_lookback = 10
delayed_days = 2
train_rets  = series_to_nfeatures(ret1[:, 0:train_days], num_lookback,
                                  np.nanmean(ret1[:, 0:train_days], 0))
train_label = np.sign(ret1[:, delayed_days:train_days+delayed_days]).flatten()
train_label[train_label==0] = np.nan
cal_nan_index = lambda x, y: np.logical_or(np.any(np.isnan(x), 1), np.isnan(y))
nan_index = cal_nan_index(train_rets, train_label)
train_rets  = train_rets[~nan_index]
train_label = train_label[~nan_index]

test_rets  = series_to_nfeatures(ret1[:, train_days:-delayed_days], num_lookback,
                                 np.nanmean(ret1[:, train_days:-delayed_days], 0))
test_label = np.sign(ret1[:, train_days+delayed_days:]).flatten()
test_label[test_label==0] = np.nan
nan_index = cal_nan_index(test_rets, test_label)
test_rets  = test_rets[~nan_index]
test_label = test_label[~nan_index]

## naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(train_rets, train_label)

pred_ret = gnb.predict(test_rets)

## decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=100)
clf = clf.fit(train_rets, train_label)
pred_ret = clf.predict(test_rets)

## metrics for prediction quality
import sklearn.metrics
report = sklearn.metrics.classification_report(test_label,
                                               pred_ret)
print report
##
from sklearn.externals.six import StringIO     
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("return_tree_allrets.pdf")                         