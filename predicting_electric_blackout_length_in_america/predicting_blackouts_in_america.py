import pandas as pd
import numpy as np
import os
from datetime import datetime, time
import seaborn as sns #statistical data visualization library based on matplotlib. seaborn provides a high-level interface for creating attractive and informative statistical graphics
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn import datasets, linear_model
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, ElasticNetCV, Lasso
# Lasso regression is a type of linear regression with L1 regularization, which can be used for feature selection and regularization.
# ElasticNet is used for linear regression with combined L1 and L2 priors as regularizers. ElasticNetCV performs cross-validated optimization to find the best combination of the alpha (regularization strength) and l1_ratio (mixing parameter) parameters for Elastic Net regression.
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import randint
from sklearn.ensemble import  AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor 
# ADABoost is used for building an ensemble of decision tree regressors through adaptive boosting. Adaptive boosting (AdaBoost) is a machine learning algorithm that combines the predictions of multiple base estimators (in this case, decision tree regressors) to improve the overall accuracy of the model
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost.sklearn import XGBRegressor # popular gradient boosting library
import lightgbm as lgbm
#LGBM stands for Light Gradient Boosting Machine, and it is implemented in the lightgbm Python package. LightGBM is a gradient boosting framework developed by Microsoft that uses tree-based learning algorithms. It is designed to be efficient and scalable, making it a popular choice for large datasets and high-dimensional features.

#target variable: OUTAGE.DURATION

############ Load Data
# # Get the current working directory
current_directory = os.getcwd()

# Path to your Excel file
excel_file_path = current_directory+'/outage.xlsx'

# # Read data from Excel file, specifying header row and skipping rows before the data
# outages = pd.read_excel(excel_file_path, header=6, skiprows=1, usecols='B:BE')

#try 2
with pd.ExcelFile(excel_file_path, engine='openpyxl') as xls:
    # Read the header row from cells B6:BE6
    header = pd.read_excel(xls, header=None, usecols="B:BE", nrows=1, skiprows=5).iloc[0]
    # Read the data from cells B8:BE1541, using the header row extracted above
    df = pd.read_excel(xls, header=None, usecols="B:BE", skiprows=7, names=header)

# Print the data
df.head()


############ data cleaning
# # convert OUTAGE.START.TIME (string of form 6:38:00 PM) to minute of the day
# def time_to_minute_of_day(time_obj):
#     if isinstance(time_obj, datetime):
#         minute_of_day = time_obj.hour * 60 + time_obj.minute
#         if time_obj.hour >= 12:
#             minute_of_day += 12 * 60
#         return minute_of_day
#     else:
#         return None

# data['MINUTE.OF.DAY']= data['OUTAGE.START.TIME'].apply(time_to_minute_of_day)

# nevermind... OUTAGE.START.TIME isn't 5:00:00 PM (as it appears in Excel file) but rather 17:00:00
def time_to_minute_of_day(time_obj):
    if isinstance(time_obj, time):
        minute_of_day = time_obj.hour * 60 + time_obj.minute
        return minute_of_day
    else:
        # Handle invalid time objects
        print(f"Invalid time object: {time_obj}")
        return None

df['MINUTE.OF.DAY']= df['OUTAGE.START.TIME'].apply(time_to_minute_of_day)

df[['MINUTE.OF.DAY', 'OUTAGE.START.TIME']].head()

# now we no longer need 'OUTAGE.START.TIME'
df = df.drop('OUTAGE.START.TIME', axis=1)

# drop redundant variables (e.g., we don't need US State AND the State abbreviation)
redundant_variables = [
                       'POSTAL.CODE',
                       'CLIMATE.REGION',
                       'PCT_WATER_TOT', # We already have 'PCT_LAND' (PCT_LAND + PCT_WATER_TOT = 100%)
                       'U.S._STATE', # Let's use NERC.REGION for the "geographic location" (mostly because I don't want to make a dummy for every state)
                       'OUTAGE.START.DATE', # we already have month (and year) which are probably good enough
                       'OUTAGE.RESTORATION.DATE', # perhaps interesting but 
                       'OUTAGE.RESTORATION.TIME'
                       ]

df = df.drop(columns=redundant_variables)

# we're just gonna use OUTAGE.DURATION as our target variables. These also might be fun to target, but maybe for another time
alternative_target_variables =  ['DEMAND.LOSS.MW',
                                 'CUSTOMERS.AFFECTED'] 

df = df.drop(columns=alternative_target_variables)

# hurricane names won't be helpful and the cause category detail is too granular
realistically_not_gonna_use = [
    'HURRICANE.NAMES',
    'CAUSE.CATEGORY.DETAIL'
]

df = df.drop(columns=realistically_not_gonna_use)

# Which columns have missing values
print([col for col in df.columns if df[col].isnull().any()])

# Given some records have blanks for our target variable, OUTAGE.DURATION, let's remove those
df = df[df['OUTAGE.DURATION'].notna()]

# check which columns are null again
print([col for col in df.columns if df[col].isnull().any()])

# all of the columns with missing data are continuous, so let's impute the median value in for missing values
# Impute missing values with the median of each column
for column in df.columns:
    if df[column].isnull().any():  # Check if the column has missing values
        median_value = df[column].median()  # Calculate the median of the column
        df[column].fillna(median_value, inplace=True)  # Fill missing values with the median

# Let's make dummy variables for all of our categorical variables (one-hot encoding)
df = pd.get_dummies(df, columns=['NERC.REGION', 'CLIMATE.CATEGORY', 'CAUSE.CATEGORY'])

# We have one record where the NERC.REGION is "FRCC, SERC." Let's disperse this across the dummies for both NERC.REGION_FRCC and NERC.REGION_SERC (instead of having a dummy for NERC.REGION_FRCC, SERC)
# Replace 0s in "NERC.REGION_FRCC" and "NERC.REGION_SERC" with 1s where "NERC.REGION_FRCC, SERC" equals 1
df.loc[df["NERC.REGION_FRCC, SERC"] == 1, ["NERC.REGION_FRCC", "NERC.REGION_SERC"]] = 1

# double check it worked
filtered_df = df[df["NERC.REGION_FRCC, SERC"] == 1]

# Select and display the specified columns
filtered_df[["NERC.REGION_FRCC", "NERC.REGION_SERC", "NERC.REGION_FRCC, SERC"]]

# drop filtered_df because it just just to check our operations worked
del filtered_df

# We also no longer need the column "NERC.REGION_FRCC, SERC"
df = df.drop(columns=["NERC.REGION_FRCC, SERC"])


############ data exploration
# Correlation matrix
corr_mat = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_mat, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

# Which variables are most correlated with OUTAGE.DURATION
corr_mat["OUTAGE.DURATION"].sort_values(ascending=False)

# Because Python is truncating my results, let's show the entire thing
# Show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Display the sorted correlation values
sorted_corr = corr_mat["OUTAGE.DURATION"].sort_values(ascending=False)
print(sorted_corr)

# Reset the display options to their default values (if needed) to avoid affecting subsequent output
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')


# What are the most important variables per a random forest model?
X = df.loc[:, (df.columns != 'OUTAGE.DURATION') & (df.columns != 'OBS') ]
Y = df["OUTAGE.DURATION"]

random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
random_forest.fit(X, Y)

# Get feature importances
feature_importances = random_forest.feature_importances_

# Create a DataFrame to store feature names and their corresponding importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top N most important features
top_n = 10  # Change this number to display a different number of top features
print(f"Top {top_n} most important features:")
print(feature_importance_df.head(top_n))

# what times of day have the longest outages?
# Create a new df representing the 60-minute interval
time_of_day_summary = df[['MINUTE.OF.DAY', 'OUTAGE.DURATION']]
time_of_day_summary['hour_of_day'] = (time_of_day_summary['MINUTE.OF.DAY'] - 1) // 60 + 1
time_of_day_summary.groupby('hour_of_day')['OUTAGE.DURATION'].agg(['mean', 'median', 'count']).reset_index()

# Seems they MOST often occur starting in the early afternoon
# but the median is shortest when starting in late morning 

############ Standardize the variables
variables_to_exclude_from_standardizing = [
    'OBS',
    'YEAR',
    'MONTH',
    'OUTAGE.DURATION', # target variable
    'NERC.REGION_ECAR', #the rest of these are dummy variables
    'NERC.REGION_FRCC',
    'NERC.REGION_HECO',
    'NERC.REGION_HI',
    'NERC.REGION_MRO',
    'NERC.REGION_NPCC',
    'NERC.REGION_PR',
    'NERC.REGION_RFC',
    'NERC.REGION_SERC',
    'NERC.REGION_SPP',
    'NERC.REGION_TRE',
    'NERC.REGION_WECC',
    'CLIMATE.CATEGORY_cold',
    'CLIMATE.CATEGORY_normal',
    'CLIMATE.CATEGORY_warm',
    'CAUSE.CATEGORY_equipment failure',
    'CAUSE.CATEGORY_fuel supply emergency',
    'CAUSE.CATEGORY_intentional attack',
    'CAUSE.CATEGORY_islanding',
    'CAUSE.CATEGORY_public appeal',
    'CAUSE.CATEGORY_severe weather',
    'CAUSE.CATEGORY_system operability disruption'
]

columns_to_standardize = df.columns.difference(variables_to_exclude_from_standardizing)

# Create a StandardScaler object
scaler = StandardScaler()

# Standardize only the selected columns
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])


############ Models
#for displaying scores at end
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


############ Simple linear regression
lm = LinearRegression()
X = df[['YEAR', 
        'CAUSE.CATEGORY_fuel supply emergency',
        'CAUSE.CATEGORY_severe weather',
        'PCT_LAND',
        'UTIL.CONTRI',
        'MINUTE.OF.DAY']]
Y = df["OUTAGE.DURATION"]

model = sm.OLS(Y, X).fit()

#cross validation
lin_scores = cross_val_score(lm, X, Y,
                             scoring="neg_mean_squared_error", cv=5)
lin_rmse_scores = np.sqrt(-lin_scores)

#Display scores
display_scores(lin_rmse_scores)

print(model.summary())
print("The average cross validation score is ", np.mean(lin_rmse_scores))


############ Regression Tree
X = df.loc[:, (df.columns != 'OUTAGE.DURATION') & (df.columns != 'OBS') ]
Y = df["OUTAGE.DURATION"]
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X, Y)

# Select cross validation with 5 folds
scores = cross_val_score(tree_reg, X, Y,
                         scoring="neg_mean_squared_error", cv=5)
# The scores the negative MSE, so we use "minus" in the square root. This is just a feature of cross-validation
tree_rmse_scores = np.sqrt(-scores)

#Display scores
display_scores(tree_rmse_scores)
print("The average cross validation score is ", np.mean(tree_rmse_scores))


############ Randomized Grid Search of Forest Regression
param_distribs = {
        'n_estimators': randint(low=1, high=20),
        'max_features': randint(low=1, high=20),
    }

forest_reg = RandomForestRegressor(random_state=42)
# Set the number of iterations as 5
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=30, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X, Y)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#GridSearchCV
# n_estimators is the number of trees in the forest. max_features is the largest number of features in a forest,
# boostrap is the estimation of forest average by randomly dropping some trees. 
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [19], 'max_features': [7]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [19], 'max_features': [7]},
  ]

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X, Y)

grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_



############ SK Learn / split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Let's make an empty DF to append all of the 
sk_learn_model_scores = pd.DataFrame(columns=['model', 'rmse'])



############ REGULARIZATION WITH ELASTIC NET
# Set parameters to iterate over
alphas = [0.0005]
l1ratio = [0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]
# Model with iterative fitting 
elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)

# Fit the model to the data
estc_reg = elastic_cv.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = estc_reg.predict(X_test)
print("ElasticRegressor RMSE:", sqrt(mean_squared_error(y_test, y_pred)))

sk_learn_model_scores = sk_learn_model_scores.append(
    {'model': 'ElasticRegressor', 
     'rmse': sqrt(mean_squared_error(y_test, y_pred))},
     ignore_index=True
    )



############ REGULARIZATION WITH LASSO
# Set parameters to iterate over
parameters= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}

# Instantiate reg for gridsearch
lasso=Lasso()
# Conduct the gridsearch
lasso_reg = GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)

# Instantiate new lasso reg with best params
lasso_reg = Lasso(alpha= 0.0009)

# Fit the model to the data
lasso_reg.fit(X_train,y_train)

# Predict on the test set from our training set
y_pred = lasso_reg.predict(X_test)
print("LassoRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

sk_learn_model_scores = sk_learn_model_scores.append(
    {'model': 'LassoRegressor', 
     'rmse': sqrt(mean_squared_error(y_test, y_pred))},
     ignore_index=True
    )


############ ADA BOOST
# Grid search for best params
param_grid = {
 'n_estimators': [50, 100, 200],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
 'loss' : ['linear', 'square', 'exponential']
 }

# Instantiate reg for gridsearch
ab_reg = AdaBoostRegressor()

# Conduct the gridsearch
grid_search = GridSearchCV(estimator = ab_reg, param_grid = param_grid, cv = 4, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

# Create a random forest with best parameters
ab_reg = AdaBoostRegressor(learning_rate =1, loss = 'exponential', n_estimators =  50, random_state= 12)

# Fit the model to the data
ab_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred_ab = ab_reg.predict(X_test)
print("AdaBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_ab)))

sk_learn_model_scores = sk_learn_model_scores.append(
    {'model': 'AdaBoostRegressor', 
     'rmse': sqrt(mean_squared_error(y_test, y_pred_ab))},
     ignore_index=True
    )


############ XGBOOST
# Grid search for best params
param_grid = {'max_depth':[3,4],
          'learning_rate':[0.01,0.03],
          'min_child_weight':[1,3],
          'reg_lambda':[0.1,0.5],
          'reg_alpha':[1,1.5],      
          'gamma':[0.1,0.5],
          'subsample':[0.4,0.5],
         'colsample_bytree':[0.4,0.5],
}

# Instantiate reg for gridsearch
reg = XGBRegressor()

# Conduct the gridsearch
grid_search = GridSearchCV(estimator = reg, param_grid = param_grid,
                          cv = 4, n_jobs = -1, verbose = True)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# Create a regressor with best parameters
xgb_reg = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3,min_child_weight=0, 
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', 
nthread=-1, scale_pos_weight=1, seed=27,reg_alpha=0.00006)

# Fit the model to the data
xgb_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = xgb_reg.predict(X_test)
print("XGBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

sk_learn_model_scores = sk_learn_model_scores.append(
    {'model': 'XGBoostRegressor', 
     'rmse': sqrt(mean_squared_error(y_test, y_pred))},
     ignore_index=True
    )

############ LIGHTGBM
# Instantiate reg
lgbm_reg = lgbm.LGBMRegressor(
    objective='regression',
    num_leaves=4,
    learning_rate=0.01,
    n_estimators=5000,
    max_bin=200,
    bagging_fraction=0.75,
    bagging_freq=5,
    bagging_seed=7,
    feature_fraction=0.2,
    feature_fraction_seed=7,
    verbose=-1,
    #min_data_in_leaf=2,
    #min_sum_hessian_in_leaf=11
)

# Fit the model to the data
lgbm_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = lgbm_reg.predict(X_test)
print("LGBMRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))


sk_learn_model_scores = sk_learn_model_scores.append(
    {'model': 'LGBMRegressor', 
     'rmse': sqrt(mean_squared_error(y_test, y_pred))},
     ignore_index=True
    )