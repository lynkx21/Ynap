
# coding: utf-8

# # YOOX Net-A-Porter - Technical Exercise
# 
# The purpose of this exercise is to try to identify customers at risk of lapse in order to take actoin before they stop to shop.
# 
# #### This is the code for the final test. 
# I'll skip most of the part of the analysis that can be found on the document.
# Here I'll import the final model I've saved there and it will run on the data provided here.
# 
# Given all the considerations done in the document, I'm setting the code as if there are two input datasets, one with the customer table *account* and one with the *transactions* with the same columns as in the beginning.
# 
# Please check the folder where I'm loading the data (*ynap_data*), thanks.

# In[ ]:


# importing libraries
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings(action="ignore")

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.externals import joblib
import xgboost as xgb

# data in .../ynap_data
data_path = os.path.join(os.getcwd(), 'ynap_data')
df_acc = pd.read_csv(os.path.join(data_path, 'account_test.csv'), sep=',')
df_trans = pd.read_csv(os.path.join(data_path, 'transactions_test.csv'), sep=',')


# In[ ]:


# feature engineering on transactions
df_trans['order_date'] = df_trans['order_date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
df_trans['order_quarter'] = df_trans['order_date'].apply(
    lambda x: str(x.date().year) + str((x.date().month-1)//3+1).zfill(2))
df_trans['item_returned'] = df_trans['net_spend'].apply(lambda x: 1 if x == 0 else 0)
df_trans['item_bought'] = df_trans['net_spend'].apply(lambda x: 0 if x == 0 else 1)


# In[ ]:


# column finder is a function that finds all the columns of a dataframe for a given substring
def column_finder(df, col, method=2):
    if method == 1:
        # method 1:
        cols_as_string = ' '.join(df.columns.values)
        cols_found = list(re.findall(col + '.*?\ ', cols_as_string))
        cols_found = [x.strip(' ') for x in cols_found]
    elif method == 2:
        # method 2:
        cols_found = []
        for col_elem in df.columns.values:
            if col in col_elem:
                cols_found.append(col_elem)
    else:
        raise ValueError
            
    return cols_found

# get_piv_counts_and_sums performs all the aggregations and the calculation of the average metrics
# if necessary, it performs a transposition of the table.
def get_piv_counts_and_sums(df, grouping_columns='customer_id'):
    
    dist_count_cols = ['order_id', 'product_id', 'product_type_id', 'designer_id']
    count_cols = ['var1', 'var2']
    sum_cols = ['gross_spend', 'net_spend', 'item_bought', 'item_returned', 'var1', 'var2']
    
    # maybe not necessary
    if type(grouping_columns) != list:
        glist = []
        glist.append(grouping_columns)
    else:
        glist = grouping_columns
   
    get_dist_counts = df.groupby(glist)[dist_count_cols].nunique().reset_index()
    get_counts = df.groupby(glist)[count_cols].count().reset_index()
    get_sums = df.groupby(glist)[sum_cols].sum().reset_index()
    get_sums = get_sums.rename(columns = {'var1': 'var1_sum', 'var2': 'var2_sum'})
    get_sums['quote_spend_returned'] = (get_sums['gross_spend'] - 
                                        get_sums['net_spend']) / get_sums['gross_spend']
    
    counts_and_sums = pd.merge(left=get_dist_counts, right=get_sums, how='inner', on=glist)
    counts_and_sums = pd.merge(left=counts_and_sums, right=get_counts, how='inner', on=glist)
    counts_and_sums['quote_var1'] = counts_and_sums['var1_sum'] / counts_and_sums['var1']
    counts_and_sums['quote_var2'] = counts_and_sums['var2_sum'] / counts_and_sums['var2']
    counts_and_sums.drop(['var1_sum', 'var1', 'var2_sum', 'var2'], axis=1, inplace=True)
    
    if len(glist) > 1:
        glist.pop(glist.index('customer_id'))
        if len(glist) == 1:
            glist = glist[0]
        
        cs_pivot = counts_and_sums.pivot(index='customer_id', columns=glist).fillna(0)
        csp_new_cols = []
        for col in cs_pivot.columns.get_values():
            csp_new_cols.append('_'.join(col))
                
        cs_pivot.columns = csp_new_cols
        cs_pivot = cs_pivot.reset_index()
        
        for val in counts_and_sums[glist].unique():
            #  relative metrics for time based tables
            cs_pivot['ns_per_order_' + str(val)] = cs_pivot['net_spend_' + str(val)] / cs_pivot['order_id_' + str(val)]
            cs_pivot['gs_per_order_' + str(val)] = cs_pivot['gross_spend_' + str(val)] / cs_pivot['order_id_' + str(val)]
            cs_pivot['ib_per_order_' + str(val)] = cs_pivot['item_bought_' + str(val)] / cs_pivot['order_id_' + str(val)]
            cs_pivot['ir_per_order_' + str(val)] = cs_pivot['item_returned_' + str(val)] / cs_pivot['order_id_' + str(val)]
            cs_pivot['ns_per_ib_' + str(val)] = cs_pivot['net_spend_' + str(val)] / cs_pivot['item_bought_' + str(val)]
            cs_pivot['gs_per_item_' + str(val)] = cs_pivot['gross_spend_' + str(val)] / (cs_pivot['item_bought_' + str(val)]
                                         + cs_pivot['item_returned_' + str(val)])
            
        return cs_pivot.fillna(0)
    
    else:
        # relative metrics for tot table
        counts_and_sums['ns_per_order'] = counts_and_sums['net_spend'] / counts_and_sums['order_id']
        counts_and_sums['gs_per_order'] = counts_and_sums['gross_spend'] / counts_and_sums['order_id']
        counts_and_sums['ib_per_order'] = counts_and_sums['item_bought'] / counts_and_sums['order_id']
        counts_and_sums['ir_per_order'] = counts_and_sums['item_returned'] / counts_and_sums['order_id']
        counts_and_sums['ns_per_ib'] = counts_and_sums['net_spend'] / counts_and_sums['item_bought']
        counts_and_sums['gs_per_item'] = counts_and_sums['gross_spend'] / (counts_and_sums['item_bought']
                          + counts_and_sums['item_returned'])
            
    return counts_and_sums.fillna(0)

quarterly_stats = get_piv_counts_and_sums(df_trans, ['customer_id', 'order_quarter'])


# In[ ]:


# combining the account table with the aggregations
df_qua = pd.merge(left=df_acc, right=quarterly_stats, on='customer_id', how='inner')

print('quarterly aggregations dataframe dimension: ({:,}, {:,})'.format(df_qua.shape[0], df_qua.shape[1]))


# In[ ]:


# transformation to factors
df_qua['var3'] = df_qua['var3'].astype(float).astype('category')
df_qua['var5'] = df_qua['var5'].astype(float).astype('category')
df_qua['var6'] = df_qua['var6'].astype(float).astype('category')


# In[ ]:


# log transformation of columns
def boxcox_func(x, l=1):
    return np.log(x + l)

def boxcox_trans(df, var_list):
    for var in var_list:
        if df[var].max() <= 1:
            df[var] = df[var].apply(boxcox_func, l=0.001)
        else:
            df[var] = df[var].apply(boxcox_func)
    return df

skewed_cols = pd.read_csv('log_trans_cols.csv')
skewed_cols = list(skewed_cols['col_name'])

df_qua_log = boxcox_trans(df_qua, skewed_cols)


# In[ ]:


# helper to find the columns on a dataframe and remove them
def column_remover(df, col_list, noprint=False):    
    if type(col_list) != list:
        cl = []
        cl.append(col_list)
        col_list =cl
    
    col_to_remove = []
    for col in col_list:
        #print(col)
        col_to_remove.extend(column_finder(df, col, method=2))
    
    df.drop(col_to_remove, axis=1, inplace=True)
    if not noprint:
        print(col_to_remove)
        

columns_to_remove = ['product_id', 'designer_id', 'gross_spend', 'item_bought', 'item_returned',
                     'ns_per_order', 'ir_per_order', 'gs_per_item', '_1989', '_199001', '_199002']
column_remover(df_qua_log, columns_to_remove, noprint=True)


# In[ ]:


# make dummies out of categorical variables
dummy_idx = np.where(df_qua_log.dtypes == 'category')[0]
df_dummies = pd.get_dummies(df_qua_log.iloc[:, dummy_idx])
df = pd.concat([df_qua_log.drop(df_qua_log.iloc[:, dummy_idx].columns.values, axis=1), df_dummies], axis=1)


# In[ ]:


# load model and cols to consider in the model
model_cols = pd.read_csv('model_cols_list.csv')
model_cols = list(model_cols['col_name'])

filename = 'ynap_best_model.sav'
svc_best = joblib.load(filename)
svc_best


# In[ ]:


svc_best.predict(df[model_cols])


# In[ ]:


# recall measure
def right_classification(y_true, y_pred):
    cf = confusion_matrix(y_true, y_pred)
    return cf[1, 1] / cf[1, :].sum()

right_classification(df['lapsed_next_period'], svc_best.predict(df[model_cols]))

