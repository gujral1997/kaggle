
# coding: utf-8

# In[130]:




import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[131]:


train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train.columns


# In[132]:


X_full = pd.concat([train.drop('Survived', axis = 1), test], axis = 0)


# In[133]:


X_full.shape


# In[134]:


X_full.drop('PassengerId', axis = 1, inplace=True)


# In[135]:


X_full.isnull().sum()


# In[136]:


(X_full.Age.isnull() & X_full.Cabin.isnull()).sum()


# In[137]:


train.Survived.mean()


# In[138]:


train.Cabin.notnull().mean()


# In[139]:


(train.Cabin.isnull() & (train.Survived == 0)).mean()


# In[140]:


selector = (train.Cabin.isnull() & train.Age.isnull())

train[selector].Survived.mean()


# In[141]:


X_full['Nulls'] = X_full.Cabin.isnull().astype('int') + X_full.Age.isnull().astype('int')


# In[142]:


X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0] # this captures the letter

# this transforms the letters into numbers
cabin_dict = {k:i for i, k in enumerate(X_full.Cabin_mapped.unique())} 
X_full.loc[:, 'Cabin_mapped'] = X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)


# In[143]:


cabin_dict


# In[144]:


X_full.columns
X_full.drop(['Age', 'Cabin'], inplace = True, axis = 1)
fare_mean = X_full[X_full.Pclass == 3].Fare.mean()

X_full['Fare'].fillna(fare_mean, inplace = True)


# In[145]:


X_full.isnull().sum()


# In[146]:


X_full[X_full.Embarked.isnull()]


# In[147]:


X_full[X_full['Pclass'] == 1].Embarked.value_counts()


# In[148]:


X_full['Embarked'].fillna('S', inplace = True)


# In[149]:


X_full.isnull().sum()


# In[150]:


X_full.drop(['Name', 'Ticket'], axis = 1, inplace = True)


# In[151]:


X_dummies = pd.get_dummies(X_full, columns = ['Sex', 'Nulls', 'Cabin_mapped', 'Embarked'], drop_first= True)


# In[152]:


X_dummies.dtypes


# In[153]:


X = X_dummies[:len(train)]; new_X = X_dummies[len(train):]
y = train.Survived


# In[154]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = .3,
                                                    random_state = 5,
                                                   stratify = y)


# In[155]:


from sklearn.ensemble import RandomForestClassifier


# In[156]:


rf = RandomForestClassifier()


# In[157]:


rf.fit(X_train, y_train)


# In[158]:


rf.score(X_test, y_test)


# In[159]:


from xgboost import XGBClassifier


# In[160]:


xgb = XGBClassifier()


# In[161]:


xgb.fit(X_train, y_train)


# In[162]:


xgb.score(X_test, y_test)


# In[163]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)
lg.score(X_test, y_test)


# In[165]:




import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': range(8, 20),
    'max_depth': range(6, 10),
    'learning_rate': [.4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1]
}

# Instantiate the regressor: gbm
gbm = XGBClassifier(n_estimators=10)

# Perform random search: grid_mse
while True:
	xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = gbm, scoring = "accuracy", 
                                    verbose = 1, n_iter = 100, cv = 4)


	# Fit randomized_mse to the data
	xgb_random.fit(X, y)

	# Print the best parameters and lowest RMSE
	# print("Best parameters found: ", xgb_random.best_params_)
	#print("Best accuracy found: ", xgb_random.best_score_)
	if(xgb_random.best_score_>0.83):
		break


# In[166]:


xgb_pred = xgb_random.predict(new_X)


# In[167]:


submission = pd.concat([test.PassengerId, pd.DataFrame(xgb_pred)], axis = 'columns')


# In[168]:


submission.columns = ["PassengerId", "Survived"]


# In[169]:


submission.to_csv('titanic_submission.csv', header = True, index = False)

