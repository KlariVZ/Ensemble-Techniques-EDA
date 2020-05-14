#!/usr/bin/env python
# coding: utf-8

# ## PORTEGUESE BANK
# 
# #### CUSTOMER DETAILS - SELLING LONG TERM DEPOSIT
# #### IDENTIFY POTENTIAL CUSTOMERS
# #### INCREASE HIT RATIO

# In[1]:


# Data dictionary

# **Bank client data**
# * 1 - age 
# * 2 - job : type of job 
# * 3 - marital : marital status
# * 4 - education 
# * 5 - default: has credit in default? 
# * 6 - housing: has housing loan? 
# * 7 - loan: has personal loan?
# * 8 - balance in account

# **Related to previous contact**
# * 8 - contact: contact communication type
# * 9 - month: last contact month of year
# * 10 - day_of_week: last contact day of the week
# * 11 - duration: last contact duration, in seconds*

# **Other attributes**
# * 12 - campaign: number of contacts performed during this campaign and for this client
# * 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign
# * 14 - previous: number of contacts performed before this campaign and for this client
# * 15 - poutcome: outcome of the previous marketing campaign

# **Output variable (desired target):has the client subscribed a term deposit?**


# In[2]:


### ENABLE PLOTTING GRAPHS IN JUPYTER
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


### LIBRARIES

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics


# In[4]:


### LOAD FILE (CSV FILE) FROM LOCAL DIRECTORY - pd.read_csv

bank_df = pd.read_csv("bank-full.csv")


# In[5]:


bank_df.head()


# In[6]:


bank_df.info()


# In[7]:


### DROP THE DURATION VARIABLE
### THE DURATION VARIABLE IS UNKNOWN WHICH COULD NEGATIVELY EFFECT THE ACCURACY OF THE PREDICTIVE MODEL
### DURATION VARIABLE SHOULD ONLY BE USED AS A BENCHMARK IN THIS CASE

bank_df.drop(['duration'], inplace=True, axis=1)


# In[8]:


### CONVERT CATEGORICAL VARIABLES TO NUMERIC VARIABLES

bank_df['day']=bank_df['day'].astype('category')
bank_df['Target']=bank_df['Target'].astype('category')


# In[9]:


#### EDA ####


# In[10]:


#### UNIVARIATE ANALYSIS


# In[11]:


### BOXPLOT
sns.boxplot(x=bank_df['age'], data=bank_df)


# In[12]:


# OUTLIERS PRESENT IN AGE
# MEDIAN AGE AROUND 40
# ACCORDING TO THE BOXPLOT THERE ARE CUSTOMERS ABOVE THE AGE OF 90, THIS IS POSSIBLY A FAULT IN THE DATA


# In[13]:


### HISTOGRAMS - PAIR PLOTS
sns.pairplot(bank_df)


# In[14]:


# ALL VARIABLES, EXCLUDING AGE, SEEM TO BE SKEWED


# In[15]:


#### UNI-VARIATE ANALYSIS ####


# In[16]:


#### COUNTPLOT
#### VALUE COUNT: CATEGORICAL VARIABLES


# In[17]:


bank_df['job'].value_counts()


# In[18]:


sns.countplot(bank_df['marital'])


# In[19]:


sns.countplot(bank_df['education'])


# In[20]:


sns.countplot(bank_df['default'])


# In[21]:


sns.countplot(bank_df['housing'])


# In[22]:


sns.countplot(bank_df['loan'])


# In[23]:


sns.countplot(bank_df['contact'])


# In[24]:


sns.countplot(bank_df['poutcome'])


# In[25]:


sns.countplot(bank_df['Target'])


# In[26]:


bank_df['Target'].value_counts(normalize=True)


# In[27]:


# RESPONSE RATE LOW: 11.6%
# ACCURACY WILL PROVE TO BE AN UNRELIABLE MODEL FOR PERFORMANCE MEASUREMENT
# NOTE FALSE NEGATIVE: FN CUSTOMER WILL POTENTIALLY SUBSCRIBE FOR LOAN
# RECALL IS MOST RELIABLE MODEL PERFORMANCE MEASURE


# In[28]:


#### BIVARIATE ANALYSIS ####


# In[29]:


### GROUP NUMERICAL VARIABLES: MEAN FOR Y VARIABLE CLASSES
np.round(bank_df.groupby(["Target"]).mean() ,1)


# In[30]:


# MEAN: HIGHER FOR CUSTOMERS SUBSCRIBE COMPARED TO THOSE WHO DON'T
# DAYS PAST AFTER LAST CONTACT HIGHER FOR SUBSCRIBED CUSTOMERS
# HIGHER CONTACTS PERFORMED BEFORE CAMPAIGN FOR CUSTOMERS WHO SUBSCRIBE


# In[31]:


# SUBSCRIPTION TENDENCY: CUSTOMERS WITH HIGHER BALANCE
# SUBSCRIPTOPN TENDENCY: FREQUENT CONTACT BEFORE CAMPAIGN


# In[32]:


#### BI-VARIATE ANALYSIS: CROSSTAB ####


# In[33]:


pd.crosstab(bank_df['job'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[34]:


# HIGHEST CONVERSION: STUDENTS 28%
# LOWEST CONVERSION: BLUE-COLLAR 7%


# In[35]:


pd.crosstab(bank_df['marital'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[36]:


pd.crosstab(bank_df['education'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[37]:


print(pd.crosstab(bank_df['default'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False ))
print(bank_df['default'].value_counts(normalize=True))


# In[38]:


### DROP DEFAULT COLUMN
bank_df.drop(['default'], axis=1, inplace=True)


# In[39]:


bank_df.columns


# In[40]:


pd.crosstab(bank_df['housing'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[41]:


pd.crosstab(bank_df['contact'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[42]:


pd.crosstab(bank_df['day'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )[0:10]


# In[43]:


pd.crosstab(bank_df['month'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[44]:


#### HIGH LEVEL FINDINGS: POINTERS TO FEATURE SELECTION ####


# In[45]:


### BINNING
### minval = MINIMUM VALUES
### maxval = MAXIMUM VALUES

def binning(col, cut_points, labels=None):
  minval = col.min()
  maxval = col.max()

  #LIST
  break_points = [minval] + cut_points + [maxval]

  #IF NO LABELS: USE DEFAULT LABELS 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #PANDAS CUT FUNCTION FOR BINNING
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin


# In[46]:


### BINNING BALANCE
cut_points = [0,500,1000, 1500,2000]
labels = ["very low","low","medium","high", "very high", "highest"]
bank_df['balance_range'] = binning(bank_df['balance'], cut_points, labels)
bank_df['balance_range'].value_counts()


# In[48]:


#BINNING CAMPAIGN
cut_points = [2,3,4]
labels = ["<=2","3","4",">4"]
bank_df['campaign_range'] = binning(bank_df['campaign'], cut_points, labels)
bank_df['campaign_range'].value_counts()


# In[49]:


bank_df.drop(['balance', 'campaign'], axis=1, inplace=True)
bank_df.columns


# In[50]:


X = bank_df.drop("Target" , axis=1)
y = bank_df["Target"]   
X = pd.get_dummies(X, drop_first=True)


# In[51]:


### TRAINING & TEST SET
### 70:30 SPLIT
### RANDOM NUMBER SEEDING (7)


test_size = 0.30 
seed = 7  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[52]:


### SHAPE OF THE TRAIN AND TEST SET
X_train.shape,X_test.shape


# In[53]:


### INSTANTIATING: DECISION TREE AS DEFAULT MODEL
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# In[55]:


### CHECK IF MODEL IS OVERFIT 
### DECISION TREE PRONE TO OVERFIT DUE TO BEING A NON-PARAMETRIC ALGORITHM
y_pred = dt_model.predict(X_test)
print(dt_model.score(X_train, y_train))
print(dt_model.score(X_test , y_test))


# In[56]:


### CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))


print(recall_score(y_test, y_pred,average="binary", pos_label="yes"))


# In[57]:


# RECALL SCORE: LOW
# IMPROVE RECALL SCORE IN MODEL


# In[58]:


clf_pruned = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_pruned.fit(X_train, y_train)


# In[59]:


#### TREE: VISUALIZATION ####


# In[65]:


### LIBRARIES

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import graphviz


# In[ ]:


feature_cols = X_train.columns
dot_data = StringIO()
export_graphviz(clf_pruned, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('bank_pruned.png')
Image(graph.create_png())


# In[68]:


### FEATURE IMPORTANCE

feat_importance = clf_pruned.tree_.compute_feature_importances(normalize=False)


feat_imp_dict = dict(zip(feature_cols, clf_pruned.feature_importances_))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')

### TOP 10 FEATURES 
feat_imp.sort_values(by=0, ascending=False)[0:10]


# In[69]:


preds_pruned = clf_pruned.predict(X_test)
preds_pruned_train = clf_pruned.predict(X_train)


# In[70]:


acc_DT = accuracy_score(y_test, preds_pruned)
recall_DT = recall_score(y_test, preds_pruned, average="binary", pos_label="yes")


# In[71]:


### STORE RESULTS
resultsDf = pd.DataFrame({'Method':['Decision Tree'], 'accuracy': acc_DT, 'recall': recall_DT})
resultsDf = resultsDf[['Method', 'accuracy', 'recall']]
resultsDf


# In[72]:


# OVERFITTING: REDUCED AFTER PRUNING
# RECALL HAS REDUCED BY A LOT


# In[73]:


#### RANDOM FOREST MODEL ####


# In[74]:


### LIBRARY
from sklearn.ensemble import RandomForestClassifier


# In[75]:


rfcl = RandomForestClassifier(n_estimators = 50)
rfcl = rfcl.fit(X_train, y_train)


# In[76]:


pred_RF = rfcl.predict(X_test)
acc_RF = accuracy_score(y_test, pred_RF)
recall_RF = recall_score(y_test, pred_RF, average="binary", pos_label="yes")


# In[77]:


### STORE RESULTS
tempResultsDf = pd.DataFrame({'Method':['Random Forest'], 'accuracy': [acc_RF], 'recall': [recall_RF]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Method', 'accuracy', 'recall']]
resultsDf
resultsDf


# In[78]:


#### ADA-BOOST ####


# In[80]:


### LIBRARY
from sklearn.ensemble import AdaBoostClassifier


# In[81]:


abcl = AdaBoostClassifier( n_estimators= 200, learning_rate=0.1, random_state=22)
abcl = abcl.fit(X_train, y_train)


# In[82]:


pred_AB =abcl.predict(X_test)
acc_AB = accuracy_score(y_test, pred_AB)
recall_AB = recall_score(y_test, pred_AB, pos_label='yes')


# In[83]:


#### STORE RESULTS
tempResultsDf = pd.DataFrame({'Method':['Adaboost'], 'accuracy': [acc_AB], 'recall':[recall_AB]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Method', 'accuracy', 'recall']]
resultsDf
resultsDf


# In[84]:


#### BAGGING CLASSIFIER ####


# In[85]:


### LIBRARY
from sklearn.ensemble import BaggingClassifier


# In[86]:


bgcl = BaggingClassifier(n_estimators=100, max_samples= .7, bootstrap=True, oob_score=True, random_state=22)
bgcl = bgcl.fit(X_train, y_train)


# In[87]:


pred_BG =bgcl.predict(X_test)
acc_BG = accuracy_score(y_test, pred_BG)
recall_BG = recall_score(y_test, pred_BG, pos_label='yes')


# In[88]:


### STORE RESULTS
tempResultsDf = pd.DataFrame({'Method':['Bagging'], 'accuracy': [acc_BG], 'recall':[recall_BG]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Method', 'accuracy', 'recall']]
resultsDf
resultsDf


# In[89]:


#### GRADIENT BOOST CLASSIFIER ####


# In[90]:


### LIBRARY
from sklearn.ensemble import GradientBoostingClassifier


# In[91]:


gbcl = GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.1, random_state=22)
gbcl = gbcl.fit(X_train, y_train)


# In[92]:


pred_GB =gbcl.predict(X_test)
acc_GB = accuracy_score(y_test, pred_GB)
recall_GB = recall_score(y_test, pred_GB, pos_label='yes')


# In[93]:


### STORE RESULTS
tempResultsDf = pd.DataFrame({'Method':['Gradient Boost'], 'accuracy': [acc_GB], 'recall':[recall_GB]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Method', 'accuracy', 'recall']]
resultsDf
resultsDf


# In[94]:


### BEST MODEL PERFORMANCE: BAGGING CLASSIFIER


# In[95]:


#### END ####


# In[ ]:




