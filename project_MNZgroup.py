####################################################################################################################################
############################################ Reading Data ############################################################################
####################################################################################################################################

import numpy as np
import pandas as pd

np.random.seed(1)

data = pd.read_stata('data.dta')
print(data.columns)

X = data.drop(columns=['year', 'month', 'year_month', 'lbw', 'very_lbw'])
y = data['lbw']

############## Normalization

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = X.iloc[:460]
scaler.fit(X_train)
X_train = scaler.transform(X_train)
y_train = y.iloc[:460]

X_test = X.iloc[460:]
X_test = scaler.transform(X_test)
y_test = y.iloc[460:]


####################################################################################################################################
############################################ Feature Selection & Decision Tree ########################################################
####################################################################################################################################

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

max_depth_range = range(2, 101)
min_samples_leaf_range = range(1, 11)

mse_scores = []
for i in max_depth_range:
    for j in min_samples_leaf_range:
        dt = DecisionTreeRegressor(max_depth=i, min_samples_leaf=j)
        mse = -1 * cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()
        mse_scores.append(mse)

best_mse = np.min(mse_scores)
best_max_depth, best_min_samples_leaf = np.unravel_index(np.argmin(mse_scores), (len(max_depth_range), len(min_samples_leaf_range)))

print("Best choice of max_depth: {}".format(max_depth_range[best_max_depth]))
print("Best choice of min_samples_leaf: {}".format(min_samples_leaf_range[best_min_samples_leaf]))
print("Best MSE: {}".format(best_mse))

best_dt = DecisionTreeRegressor(max_depth=max_depth_range[best_max_depth], min_samples_leaf=min_samples_leaf_range[best_min_samples_leaf])
best_dt.fit(X_train, y_train)

feature_importance = best_dt.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
feature_importances = feature_importances.sort_values('importance', ascending=False)
print(feature_importances.head(8))


########## Feature Importance Plot

import matplotlib.pyplot as plt

indices = np.argsort(feature_importance)[::-1]
names = [X.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(10), feature_importance[indices][:10], align='center')
plt.yticks(range(10), names[:10])
plt.xlabel("Relative Importance")
plt.ylabel("Features")
plt.show()
plt.savefig('Relative Feature Importance.png')


########## Running Decision Tree model and Prediction

top_features = list(feature_importances['feature'][:15])
X_train_top = X_train[:, [X.columns.get_loc(f) for f in top_features]]
X_test_top = X_test[:, [X.columns.get_loc(f) for f in top_features]]




best_dt_top = DecisionTreeRegressor(max_depth=max_depth_range[best_max_depth], min_samples_leaf=min_samples_leaf_range[best_min_samples_leaf])
best_dt_top.fit(X_train_top, y_train)


y_pred_dt = best_dt_top.predict(X_test_top)
mse = mean_squared_error(y_test, y_pred_dt)
print("MSE for Decision Tree: {:.4f}".format(mse))


####################################################################################################################################
######################################################## KNN ########################################################
####################################################################################################################################

from sklearn.neighbors import KNeighborsRegressor


k_range = range(1, 101)

mse_scores = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    mse = -1 * cross_val_score(knn, X_train_top, y_train, cv=10, scoring='neg_mean_squared_error').mean()
    mse_scores.append(mse)

best_k = k_range[np.argmin(mse_scores)]
print("Best choice of k: {}".format(best_k))

best_knn = KNeighborsRegressor(n_neighbors=best_k)
best_knn.fit(X_train_top, y_train)


y_pred_knn = best_knn.predict(X_test_top)

mse = mean_squared_error(y_test, y_pred_knn)
print("MSE for KNN: {}".format(mse))



####################################################################################################################################
################################################### Linear Regression ########################################################
####################################################################################################################################

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_top, y_train)
y_pred_lr = lr.predict(X_test_top)

mse = mean_squared_error(y_test, y_pred_lr)
print("MSE for Linear Regression: {}".format(mse))



####################################################################################################################################
################################################### Random Forest ########################################################
####################################################################################################################################

from sklearn.ensemble import RandomForestRegressor


num_trees_list = list(range(100, 501, 5))

mse_list = []
for i in num_trees_list:
    rf_model = RandomForestRegressor(n_estimators=i, max_depth=1000, max_features='sqrt', n_jobs=-1, oob_score=True)
    rf_model.fit(X_train_top, y_train)
    mse = (1 - rf_model.oob_score_) * np.var(y_train)
    mse_list.append(mse)

best_num_trees_idx = np.argmin(mse_list)
best_num_trees = num_trees_list[best_num_trees_idx]
best_mse = mse_list[best_num_trees_idx]

print("Best number of trees:", best_num_trees)
print("Best MSE:", best_mse)


rf_best = RandomForestRegressor(n_estimators=best_num_trees, max_depth=1000, max_features='sqrt', n_jobs=-1, oob_score=True)
rf_best.fit(X_train_top, y_train)


y_pred_rf = best_knn.predict(X_test_top)

mse = mean_squared_error(y_test, y_pred_rf)
print("MSE for Random Forest: {}".format(mse))



####################################################################################################################################
################################################### Prediction ########################################################
####################################################################################################################################


x = np.arange(116)
models = ['Decision Tree', 'KNN', 'Linear Regression', 'Random Forest', 'Test']
colors = ['purple', 'blue', 'green', 'pink', 'red']
predictions = [y_pred_dt, y_pred_knn, y_pred_lr, y_pred_rf, y_test]

fig, ax = plt.subplots(figsize=(30, 12))

for i, model in enumerate(models):
    ax.plot(x, predictions[i], label=model, c=colors[i])

ax.legend()
ax.set_xlabel('Sample Number')
ax.set_ylabel('Prediction')
ax.set_title('Model Predictions vs. Test Data')

plt.savefig('Prediction.png')
plt.show()

