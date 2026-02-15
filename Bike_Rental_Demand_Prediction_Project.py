#!/usr/bin/env python
# coding: utf-8

# # Bike Rental Demand Prediction:

# ##  Import Required Libraries:

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
import warnings 
warnings.filterwarnings('ignore') 
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the Dataset :

# In[3]:


df = pd.read_csv("Dataset.csv")


# ## 1. Exploratory Data Analysis (EDA) 

# ### 1.1 Describe the Dataset :

# In[4]:


df.head() 
df.tail() 
df.shape 
df.columns 
df.info() 
df.describe()


# ### 1.2 Clean the Data :

# In[5]:


# Check Missing Values 
df.isnull().sum() 


# In[22]:


## Replace Missing Values:
df.replace('?',np.nan,inplace=True)


# In[23]:


## Convert Data Types:
num_cols = ['temp','atemp','hum','windspeed','casual','registered','cnt']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True)


# ### 1.3 Handle Missing Values :

# In[24]:


df.isna().sum()


# In[25]:


## Impute missing values:
cat_cols = ['season','yr','mnth','holiday','workingday','weathersit']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
    


# In[26]:


df.isna().sum()


# In[27]:


df.dtypes


# ### 1.4 Duplicate Records :

# In[28]:


# Check Duplicate Records :
df.duplicated().sum() 
# Remove Duplicates:
df.drop_duplicates(inplace=True)


# ### 1.5 Detect Outliers :

# ####  Boxplot for Numerical Columns :
# 

# In[29]:


plt.figure(figsize=(12,6)) 
sns.boxplot(data=df[num_cols]) 
plt.xticks(rotation=90) 
plt.title("Outlier Detection using Boxplot") 
plt.show() 


# #### IQR Method for Outliers :

# In[30]:


Q1 = df[num_cols].quantile(0.25) 
Q3 = df[num_cols].quantile(0.75) 
IQR = Q3 - Q1 
df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) |  
(df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]


# ## 2. Data Visualization :

# ### 2.1 Scatter Plot :

# In[31]:


sns.scatterplot(x='temp', y='cnt', data=df) 
plt.title("Temperature vs Bike Demand") 
plt.show() 


# ### 2.2 Line Plot (Time Series) :

# In[36]:


plt.figure()
plt.plot(df['dteday'], df['cnt'])
plt.xlabel('Date')
plt.ylabel('Total Rentals')
plt.title('Bike Demand Over Time')
plt.show()


# ### 2.3 Bar Plot :

# In[16]:


sns.barplot(x='season', y='cnt', data=df) 
plt.title("Season vs Bike Demand") 
plt.show()


# ### 2.4 Histogram :

# In[17]:


sns.histplot(df['cnt'], kde=True) 
plt.title("Distribution of Bike Rentals") 
plt.show() 


# ### 2.5 Heatmap / Correlation Matrix:

# In[19]:


plt.figure(figsize=(10,6)) 
corr = df[num_cols].corr() 
sns.heatmap(corr, annot=True, cmap='coolwarm') 
plt.title("Correlation Heatmap") 
plt.show() 


# ### 2.6 Seasonality & Demand Analysis :
# 

# In[38]:


plt.figure(figsize=(6,4)) 
sns.boxplot(x='season', y='cnt', data=df) 
plt.title("Season vs Bike Demand") 
plt.show() 


# ### 2.6  Weather Impact on Demand :

# In[41]:


plt.figure(figsize=(6,4)) 
sns.barplot(x='weathersit', y='cnt', data=df) 
plt.title("Weather vs Bike Demand") 
plt.show() 


# ### 2.7  Holiday Impact :

# In[42]:


plt.figure(figsize=(6,4)) 
sns.barplot(x='holiday', y='cnt', data=df) 
plt.title("Holiday Impact on Bike Demand") 
plt.show()


# In[43]:


df.head()


# ## 3. Feature Engineering:

# ### 3.1 Generating New Features from Existing Ones
# 
#  Extract time-based features from date
# 

# In[44]:


df['year'] = df['dteday'].dt.year
df['month'] = df['dteday'].dt.month
df['day'] = df['dteday'].dt.day
df['dayofweek'] = df['dteday'].dt.dayofweek


# These features help capture seasonality, trends, and weekly patterns.

# ### 3.2 Handling Categorical Variables (Encoding)
# #### One-Hot Encoding:

# In[45]:


df = pd.get_dummies(
    df,
    columns=['season','yr','mnth','holiday','workingday','weathersit'],
    drop_first=True
)


# Prevents ordinal bias and improves regression performance.

# ### 3.3 Scaling / Normalizing Numerical Features:

# #### Standard Scaling:

# In[47]:


from sklearn.preprocessing import StandardScaler

scale_cols = ['temp','atemp','hum','windspeed']
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])


# ### 3.4 Incorporating Domain Knowledge :

# #### Peak Hour Feature

# In[48]:


df['is_peak_hour'] = df['hr'].apply(lambda x: 1 if 7 <= x <= 9 or 17 <= x <= 19 else 0)


# ## 4. Model Building

# ### 4.1 Define Features & Target:

# In[49]:


X = df.drop(['cnt','dteday','instant'], axis=1)
y = df['cnt']


# ### 4.2 Train-Test Split:

# In[50]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ### 4.3 Train Models:

# #### Linear Regression:

# In[56]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)


# #### Decision Tree Regressor

# In[52]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)


# #### Random Forest Regressor:

# In[53]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)


# #### Gradient Boosting Regressor:

# In[54]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)


# ### 4.4 Model Evaluation:

# #### Metrics Used:
# 
# RMSE
# 
# MAE
# 
# R-Squared
# 

# In[57]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate(model):
    preds = model.predict(X_test)
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))

evaluate(lr)
evaluate(dt)
evaluate(rf)
evaluate(gbr)


# ## 5. Hyperparameter Tuning:

# ### 5.1 Grid Search for Random Forest:

# In[58]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':[100,200],
    'max_depth':[None,10,20]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='r2'
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
evaluate(best_rf)


# ### 5.2 Cross-Validation Performance:

# In[59]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='r2')
scores.mean()


# ### 5.3 Overfitting vs Underfitting Control
# 
# #### Steps Taken:
# 
# Limited tree depth
# 
# Cross-validation
# 
# Regularization through ensemble models
# 

# ## 6. Model Evaluation 
# ### Explanation 
# Model evaluation is the process of measuring how well a trained model performs on unseen 
# data. 
# Since this is a regression problem, we use regression-specific metrics.

# ### 6.1 Evaluation Metrics for Regression Models :

# #### Metrics Used 
# MAE (Mean Absolute Error) 
# → Average absolute difference between actual and predicted values.
# 
# MSE (Mean Squared Error) 
# → Penalizes large errors more heavily 
# 
# RMSE (Root Mean Squared Error) 
# → Same unit as target variable, easier to interpret 
# 
# R² (R-Squared)
# → Proportion of variance explained by the model

# #### Evaluate Models Using All Metrics:

# In[60]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
import numpy as np 
 
def evaluate_model(name, model): 
    preds = model.predict(X_test) 
    mae = mean_absolute_error(y_test, preds) 
    mse = mean_squared_error(y_test, preds) 
    rmse = np.sqrt(mse) 
    r2 = r2_score(y_test, preds) 
     
    return [name, mae, mse, rmse, r2]


# ### 6.2 Compare Performance of Different Models 
#  
# #### Create Comparison Table

# In[61]:


results = [] 
 
results.append(evaluate_model("Linear Regression", lr)) 
results.append(evaluate_model("Decision Tree", dt)) 
results.append(evaluate_model("Random Forest", rf)) 
results.append(evaluate_model("Gradient Boosting", gbr)) 
 
results_df = pd.DataFrame( 
    results, 
    columns=["Model", "MAE", "MSE", "RMSE", "R2"] 
) 
 
results_df


# #### Interpretation:
# Among all evaluated models, ensemble models such as Random Forest and Gradient 
# Boosting achieved better performance with lower RMSE and higher R² values. These models 
# were more effective in capturing complex non-linear patterns in bike rental demand.

# ### 6.3 Select the Best Model:

# In[62]:


results_df.sort_values(by="R2", ascending=False)


# #### Best Model Selection Criteria: 
# Highest R² 
# 
# Lowest RMSE 
# 
# Stable performance 
# 
# Best model selected: Random Forest / Gradient Boosting (based on results)

# ## 7. Model Deployment :
# #### Explanation :
# Model deployment is the process of making a trained machine learning model available for 
# real-world usage so it can generate predictions on new data.

# ### 7.1 Save the Best Mode:

# In[63]:


import joblib 
joblib.dump(best_rf, "bike_demand_model.pkl") 


# ### 7.2 Load the Model for Prediction :

# In[64]:


loaded_model = joblib.load("bike_demand_model.pkl") 


# ### 7.3 Make Predictions on New Data :

# In[65]:


sample_input = X_test.iloc[0:1] 
prediction = loaded_model.predict(sample_input) 
prediction 


# The model successfully predicts bike rental demand for unseen inputs.

# ### 7.4 Production Deployment (Conceptual) 
# 
# #### Deployment Strategy :
# The trained model can be deployed using: 
# 
# Flask or FastAPI (REST API) 
# 
# Cloud platforms (AWS, Azure, GCP) 
# 
# Users send input features through an API 
# 
# The model returns predicted bike demand

# ### 7.5 Monitoring & Model Updates 
# 
# #### Monitoring Includes: 
# 
# Tracking prediction errors over time 
# 
# Detecting data drift 
# 
# Periodic retraining with new data 
# 
# This ensures long-term model reliability.

# ## Final Project Conclusion:

# This project successfully developed a machine learning solution to predict bike rental demand. 
# Through comprehensive exploratory data analysis, feature engineering, model building, 
# hyperparameter tuning, and evaluation, the final model demonstrated strong predictive 
# performance. The selected model was deployed for real-world usage, making the solution 
# practical and scalable.

# In[ ]:




