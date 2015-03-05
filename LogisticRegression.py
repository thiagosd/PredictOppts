from pandas._testing import isnull
from sklearn.svm import SVR, LinearSVC

__author__ = 'Thiago'

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot

# Load the data
dsOppts = pd.read_csv('data/AllOppts.csv', low_memory=False)

### Clean up data
# Remove Won Oppts with $0 or NULL Actual Value
dsOppts.drop(dsOppts.index[(dsOppts['StateCode'] == 1) & (dsOppts['ActualValue'] == 0)], inplace=True)
dsOppts.drop(dsOppts.index[(dsOppts['StateCode'] == 1) & (isnull(dsOppts['ActualValue']))], inplace=True)

# set today's date
todays_date = pd.to_datetime('2015-02-16', format='%Y-%m-%d')
# Mass Oppt import on this date.
crm_createdon = pd.to_datetime('2010-05-07', format='%Y-%m-%d')


# convert do datetime
dsOppts['CreatedOn'] = pd.to_datetime(pd.Series(dsOppts['CreatedOn']), format='%m/%d/%Y')
# dsOppts['CreatedOnOrdinal'] = dsOppts.apply(lambda x: x['CreatedOn'].toordinal(), axis=1)
dsOppts['ActualCloseDate'] = pd.to_datetime(pd.Series(dsOppts['ActualCloseDate']))
dsOppts['EstimatedCloseDate'] = pd.to_datetime(pd.Series(dsOppts['EstimatedCloseDate']))
dsOppts['new_projectstart'] = pd.to_datetime(pd.Series(dsOppts['new_projectstart']))
dsOppts['new_projectend'] = pd.to_datetime(pd.Series(dsOppts['new_projectend']))

# drop Oppts created on 2010-05-07. There are retroactive Oppts. Bad DaysOpen feature.
dsOppts.drop(dsOppts.index[dsOppts['CreatedOn'] == crm_createdon], inplace=True)

# split Created On in 3 features
dsOppts['CreatedOnDay'] = dsOppts.apply(lambda x: x['CreatedOn'].day, axis=1)
dsOppts['CreatedOnMonth'] = dsOppts.apply(lambda x: x['CreatedOn'].month, axis=1)
dsOppts['CreatedOnYear'] = dsOppts.apply(lambda x: x['CreatedOn'].year, axis=1)
# split Actual Close Date in 3 features
dsOppts['ActualCloseDateDay'] = dsOppts.apply(lambda x: x['ActualCloseDate'].day, axis=1)
dsOppts['ActualCloseDateMonth'] = dsOppts.apply(lambda x: x['ActualCloseDate'].month, axis=1)
dsOppts['ActualCloseDateYear'] = dsOppts.apply(lambda x: x['ActualCloseDate'].year, axis=1)


### Add new Features
# Project duration: new_projectstart - new_projectend
# isinstance(todays_date, pd.tslib.Timestamp)
# TO-DO: if new_projectend prior to new_projectstart or date is NAT (not a time), set ProjectDuration to avg ProjectDuration.
projectDurationDf = pd.DataFrame()
projectDurationDf['duration'] = dsOppts.apply(lambda x: x['new_projectend'] - x['new_projectstart'], axis=1).astype(
    'timedelta64[D]')
projectDurationDf.drop(
    projectDurationDf.index[(projectDurationDf['duration'] <= 0) | isnull(projectDurationDf['duration'])], inplace=True)
# set project duration mean
projdurationmean = projectDurationDf['duration'].mean()

dsOppts['ProjectDuration'] = dsOppts.apply(lambda x: projdurationmean if (
    isnull(x['new_projectend']) | isnull(x['new_projectstart']) | (x['new_projectend'] < x['new_projectstart'])) else (
    x['new_projectend'] - x['new_projectstart']).days, axis=1)

dsOppts['ValueDifference'] = dsOppts.apply(
    lambda x: 0 if x['ActualValue'] == 0 else (x['EstimatedValue'] - x['ActualValue']), axis=1)

# Days Open: CreatedOn - ActualCloseDate. For Open Oppts, Today's Date - CreatedOn
dsOppts['DaysOpen'] = dsOppts.apply(lambda x: (todays_date - x['CreatedOn']).days if x['StateCode'] == 0 else (
    x['ActualCloseDate'] - x['CreatedOn']).days, axis=1)
# Some DaysOpen are negative because Create On date is prior to Actual Close Date, set it to Days Open average when thats the case
dsOppts.loc[dsOppts['DaysOpen'] < 0, 'DaysOpen'] = dsOppts['DaysOpen'].mean()

# Days to Close each Stage
# not possible. no access to Audit entity.

# Number of Oppts for this Account
oppts_by_acct = dsOppts.groupby(['new_parentaccount'])['new_parentaccount'].count()
dsOppts['NumberOpptsForAcct'] = dsOppts.apply(lambda x: oppts_by_acct[x['new_parentaccount']], axis=1)

# Number of Oppts for this Account per StateCode
oppts_by_acct_statecode = dsOppts.groupby(['new_parentaccount', 'StateCode'], as_index=True)[
    'new_parentaccount', 'StateCode'].count()
# rename column
oppts_by_acct_statecode.rename(columns={'new_parentaccount': 'numberOpptsPerAcctStateCode'}, inplace=True)
# Number of Lost Oppts for this Account
dsOppts['NumberLostOpptsForAcct'] = dsOppts.apply(
    lambda x: 0 if (x['new_parentaccount'], 2) not in oppts_by_acct_statecode.index else oppts_by_acct_statecode.loc[
        x['new_parentaccount'], 2].numberOpptsPerAcctStateCode, axis=1)
# Number of Won Oppts for this Account
dsOppts['NumberWonOpptsForAcct'] = dsOppts.apply(
    lambda x: 0 if (x['new_parentaccount'], 1) not in oppts_by_acct_statecode.index else oppts_by_acct_statecode.loc[
        x['new_parentaccount'], 1].numberOpptsPerAcctStateCode, axis=1)
# Number of Open Oppts for this Account
dsOppts['NumberOpenOpptsForAcct'] = dsOppts.apply(
    lambda x: 0 if (x['new_parentaccount'], 0) not in oppts_by_acct_statecode.index else oppts_by_acct_statecode.loc[
        x['new_parentaccount'], 0].numberOpptsPerAcctStateCode, axis=1)

# Number of Oppts for this Account and Market
oppts_by_acct_market = dsOppts.groupby(['new_parentaccount', 'new_market'], as_index=True)[
    'new_parentaccount', 'new_market'].count()
# rename column
oppts_by_acct_market.rename(columns={'new_parentaccount': 'numberOpptsPerAcctMarket'}, inplace=True)
dsOppts['NumberOpptsForAcctMarket'] = dsOppts.apply(
    lambda x: oppts_by_acct_market.loc[x['new_parentaccount'], x['new_market']].numberOpptsPerAcctMarket, axis=1)

# Has BDM
dsOppts['HasBDM'] = dsOppts.apply(lambda x: 0 if pd.isnull(x['new_bdm']) else 1, axis=1)

# Has CSP
dsOppts['HasCSP'] = dsOppts.apply(lambda x: 0 if pd.isnull(x['new_csp']) else 1, axis=1)
# Has SSP - column not in the file
#dsOppts['HasSSP'] = dsOppts.apply(lambda x: 0 if pd.isnull(x['new_bdm']) else 1, axis=1)



### Separate data
#dsOppts.fillna(dsOppts.mean(skipna=True), inplace=True)
dsOppts.fillna(dsOppts.mean(skipna=True), inplace=True)

# use Won and Lost Oppts for train/test, and predict StateCode of Open Oppts
WonLostOppts = dsOppts[(dsOppts['StateCode'] == 1) | (dsOppts['StateCode'] == 2)]
OpenOppts = dsOppts[dsOppts['StateCode'] == 0]
WonLostOppts.is_copy = False
OpenOppts.is_copy = False

# change StateCode values to represent percentage. 0 = Lost. 1 = Won.
WonLostOppts.loc[WonLostOppts['StateCode'] == 2, 'StateCode'] = 0

### Prediction

# set Label
WonLostOppts_labels = WonLostOppts.filter(['StateCode'])

# set Features
WonLostOppts_features = WonLostOppts.filter(
    ['DaysOpen', 'NumberOpptsForAcctMarket', 'CreatedOnMonth', 'CreatedOnYear',
     'NumberLostOpptsForAcct', 'NumberWonOpptsForAcct',
     'new_changerequest', 'new_noofresources',
     'new_reopenedopportunity', 'new_winwireinclusion',
     'ProjectDuration', 'HasBDM', 'HasCSP'])

OpenOppts_features = OpenOppts.filter(
    ['DaysOpen', 'NumberOpptsForAcctMarket', 'CreatedOnMonth', 'CreatedOnYear',
     'NumberLostOpptsForAcct', 'NumberWonOpptsForAcct',
     'new_changerequest', 'new_noofresources',
     'new_reopenedopportunity', 'new_winwireinclusion',
     'ProjectDuration', 'HasBDM', 'HasCSP'])

# Convert Categorical features into dummy features
functionalarea_dummy_units = pd.get_dummies(dsOppts['new_functionalarea'], prefix='functionalarea')
primaryworktag_dummy_units = pd.get_dummies(dsOppts['new_primaryworktag'], prefix='primaryworktag')
billingtype_dummy_units = pd.get_dummies(dsOppts['new_billingtype'], prefix='billingtype')
projecttype_dummy_units = pd.get_dummies(dsOppts['new_projecttype'], prefix='projecttype')

WonLostOppts_features = WonLostOppts_features.join(functionalarea_dummy_units).join(primaryworktag_dummy_units).join(billingtype_dummy_units).join(projecttype_dummy_units)
OpenOppts_features = OpenOppts_features.join(functionalarea_dummy_units).join(primaryworktag_dummy_units).join(billingtype_dummy_units).join(projecttype_dummy_units)


# Train x Test data split
features_train, features_test, label_train, label_test = sklearn.cross_validation.train_test_split(
    WonLostOppts_features, WonLostOppts_labels, test_size=0.2, random_state=42)

# Remove single-dimensional entries from the shape of an array.
label_train = np.squeeze(label_train)
label_test = np.squeeze(label_test)


if __name__ == '__main__':
    ### Normalize Data
    scaler = StandardScaler()

    clf = LogisticRegression(C=100.)
    estimator_tree = [('scaler', scaler), ('tree', clf)]
    clf = Pipeline(estimator_tree)
    '''
    param_grid = {
        'tree__C': [1.0, 10.0, 100.0],
        'tree__penalty': ['l1', 'l2']
    }

    clf = GridSearchCV(clf, param_grid, scoring="accuracy", verbose=1, n_jobs=-1)
    '''
    clf = clf.fit(features_train, label_train)

    # check the accuracy on the training set
    print clf.score(features_train, label_train)
    # examine the coefficients
    print pd.DataFrame(zip(WonLostOppts_features.columns, np.transpose(clf.steps[1][1].coef_)))
    #print "Best estimator found by grid search:"
    #print clf.best_estimator_

    predictions = clf.predict(features_test)

    accuracy = accuracy_score(label_test, predictions)
    print "acc: ", accuracy
    print "precision: ", precision_score(label_test, predictions)
    print "recall", recall_score(label_test, predictions)


    # get Open Oppts
    openOppts_predictions = clf.predict(OpenOppts_features)
    proba_predictions = clf.predict_proba(OpenOppts_features)
    OpenOppts['predictions'] = openOppts_predictions
    OpenOppts['proba_lost_predictions'] = proba_predictions[:, 0]
    OpenOppts['proba_won_predictions'] = proba_predictions[:, 1]

    #OpenOppts.to_csv('data/output_LogisticRegression.csv')

    dsInternalCRMOppts = pd.read_csv('data/InternalCRMOppts3.csv', low_memory=False)
    dsInternalCRMOppts = dsInternalCRMOppts.filter(['Opportunity Name', 'Status'])
    dsInternalCRMOppts['StateCode'] = dsInternalCRMOppts.apply(lambda x: 1 if x['Status'] == 'Won' else 0, axis=1)

    matchedOppts = pd.merge(OpenOppts, dsInternalCRMOppts, left_on='Name', right_on='Opportunity Name', how='inner')

    dd = matchedOppts[matchedOppts['StateCode_y'] == matchedOppts['predictions']]
    dn = matchedOppts[matchedOppts['StateCode_y'] != matchedOppts['predictions']]
    print len(dd), len(dn)
    matchedOppts.to_csv('data/matchedOpptsLogisticRegression.csv')