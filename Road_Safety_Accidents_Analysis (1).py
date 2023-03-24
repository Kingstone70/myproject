#!/usr/bin/env python
# coding: utf-8

# # Road Safety Accidents Analysis
# ###  Wilson adejo
# ###  25-02-2023
# 
# 
# ## Contents
# 
# ### <a href='#1'>1. Read in the dataset</a>
#              
# ### <a href='#2'>2. Exploration Data Analysis</a>
# 
# ### <a href='#3'>3. Data Preprocessing & Feature Engineering</a>
#  
# ### <a href='#4'>4. Data Analysis</a>
#     
# ### <a href='#5'>5.Modelling and Evaluation</a>
#       
# 

# In[1]:


#importing the necessary  libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas_profiling as pp  - - Now depreciated
import ydata_profiling as pp
import sklearn as sk
import datetime 
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

#import geopandas
import shapely
#from shapely.geometry import Point
import missingno as msn

get_ipython().run_line_magic('matplotlib', 'inline')


# 
# ## <a id='1'>1. Read in the dataset</a>
# 
# 

# In[2]:


# Import the dataset and remove leading/trailing whitespaces 
df=pd.read_csv("Road_Safety_Accidents.csv",skipinitialspace = True)

print("The shape of data is:",(df.shape))
df.sample(3)  # sample  show the data randomly


# 
# ## <a id='2'>2. Exploration Data Analysis</a>

# This section contain the exploratory data analysis carried out to have a general overview of the dataset. A dataset with a total of 6914 rows and 35 columns was examined and analysed. It has  12 numerics and 23 categorical columns with 2.5% missing cells

# In[3]:


## Quick summary of the Road Safety accident data
df.describe()


# In[4]:


# Checking the various data type
df.info()


# In[5]:


# Using ProfileReport to have a quick overview of the data and save as output.html file 
profile = pp.ProfileReport(df) 
profile.to_file("output.html")
profile


# ## <a id='3'>3. Data Preprocessing & Feature Engineering</a>

# In[6]:


#Check the number of NA
df.isna().sum()


# In[7]:


# Drop all unnecessary columns 
#Features 'OBJECTID'  and other didn't provide any useful information about accidents that might be useful in the analysis
df.drop(["X",'Y','OBJECTID','POLICE_FORCE','LOCAL_AUTHORITY__DISTRICT_','LOCAL_AUTHORITY__HIGHWAY_','LOCATION_EASTING_OSGR','LOCATION_NORTHING_OSGR'], axis=1, inplace = True)
df.head(3)


# In[8]:


## Fill the missing  values using Last observation carried forward (LOCF)
df["JUNCTION_CONTROL"] = df["JUNCTION_CONTROL"].fillna(method ='ffill')
df["CASUALITY_TYPE"] = df["CASUALITY_TYPE"].fillna('Undefined')


# ### Handling  Date and time Format

# In[9]:


# Extract the Date, Year, time and how

df['DATE'] = pd.to_datetime(df['DATE_']).dt.date
df['Year'] = pd.DatetimeIndex(df['DATE']).year
df['TimeOnly'] = pd.to_datetime(df['TIME']).dt.time
df['Hour'] = pd.to_datetime(df['TIME']).dt.hour

df.drop(['DATE_', 'TIME'], axis=1, inplace=True)


df.head(3)


# In[10]:


# cast the Hour column to integer values
df['Hour'] = df['Hour'].astype('int')


# Assumptions:</br>
# Morning Rush from 5am to 10am --> value 1 </br>
# Office Hours from 10am to 3pm (or: 10:00 - 15:00) --> value 2 </br>
# Afternoon Rush from 3pm to 7pm (or: 15:00 - 19:00) --> value 3 </br>
# Evening from 7pm to 11pm (or: 19:00 - 23:00) --> value 4 </br>
# Night from 11pm to 5am (or: 23:00 - 05:00) --> value 5 </br>

# In[11]:


# Define a function that turns the hours into daytime group
def when_was_it(hour):
    if hour >= 5 and hour < 10:
        return "1"
    elif hour >= 10 and hour < 15:
        return "2"
    elif hour >= 15 and hour < 19:
        return "3"
    elif hour >= 19 and hour < 23:
        return "4"
    else:
        return "5"


# In[12]:


daytime_groups = {1: 'Morning: Between 5 and 10', 
                  2: 'Office Hours: Between 10 and 15', 
                  3: 'Afternoon Rush: Between 15 and 19', 
                  4: 'Evening: Between 19 and 23', 
                  5: 'Night: Between 23 and 5'}


# In[13]:


# apply this function to our temporary hour column
df['Daytime'] = df['Hour'].apply(when_was_it)
df.head()


# In[14]:


df.info()


# ## <a id='4'>4. Data Analysis</a>

# ### Grouping By Columns

# ### A. Road type with the highest number of traffic accident

# In[15]:


df_road=df.groupby('ROAD_TYPE').ACCIDENT_INDEX.count()
print(df_road)
df_road.plot(kind='bar', title = 'Road Types and Total Number of Accident')


# ### B. Accident Severity over the period

# #### Count of  Accident by  Hour  and the Severity of it
# Most accidents happened during the daytime, especially AM peak and PM peak. When it comes to night, accidents were far less but more likely to be serious.

# In[16]:


hour_df=df.groupby('Hour').ACCIDENT_INDEX.count()
hour_df


# In[17]:


plt.figure(figsize=(15,9))
sns.countplot(x='Hour', hue='ACCIDENT_SEVERITY', data=df ,palette="Set2")
plt.title('Count of Accidents by Hour', size=20, y=1.05)
plt.show()


# #### * Count and percentage of  Accident by the Severity of it

# In[18]:


Severity_class=df.groupby('ACCIDENT_SEVERITY').ACCIDENT_INDEX.count()
Severity_class


# In[19]:


plt.figure(figsize=(8, 8))
Severity_class.plot(kind='pie', y='ACCIDENT_INDEX', autopct='%1.0f%%',fontsize=10)
plt.legend(fontsize=10)
plt.show()


# In[20]:


# create a dataframe of Severity and the corresponding accident cases
severity_df = pd.DataFrame(df['ACCIDENT_SEVERITY'].value_counts()).rename(columns={'index':'ACCIDENT_SEVERITY', 'ACCIDENT_SEVERITY':'Cases'})


# In[21]:


import plotly as pt
from plotly import graph_objs as go
import plotly.express as px
fig = go.Figure(go.Funnelarea(
   # text = ["Severity - 2","Severity - 3", "Severity - 4", "Severity - 1"],
    values = severity_df.Cases,
    title = {"position": "top center", 
             "text": "<b>Percentage of Accident by the Severity</b>", 
             'font':dict(size=18,color="#7f7f7f")},
    marker = {"colors": ['#14a3ee', '#b4e6ee', 'Yellow'],
                "line": {"color": ["#e8e8e8", "wheat", "wheat", "wheat"], "width": [7, 0, 0, 2]}}
    ))

fig.show()


# #### * Count of  Traffic Accident by the Severity and road type

# In[22]:


grouped_df = df.groupby(["ROAD_TYPE", "ACCIDENT_SEVERITY"])['NUMBER_OF_CASUALTIES'].sum().reset_index()
print (grouped_df)


# In[23]:



df_source = df.groupby(['ACCIDENT_SEVERITY','ROAD_TYPE']).size().reset_index().pivot(    columns='ACCIDENT_SEVERITY', index='ROAD_TYPE', values=0)

df_source.plot(kind='bar', stacked=True, title='Accident Severity Count by Road Types', fontsize=12)


# In[24]:


grouped_df.plot(kind='bar', title = 'NUMBER_OF_CASUALTIES')
plt.ylabel("NUMBER_OF_CASUALTIES")


# In[25]:


df.groupby("ACCIDENT_SEVERITY")['NUMBER_OF_CASUALTIES'].sum().sort_values(ascending=True).plot(kind='bar',figsize = (12,7),title = 'Number of Causalties by Severity of the Accident')
plt.ylabel("Number of Casualities")


# ### Accident severity  and  number of casulaties

# In[26]:


## Accident severity  and  number of casulaties
grouped_accidents = df.groupby('ACCIDENT_SEVERITY').agg({'NUMBER_OF_CASUALTIES': ['sum', 'min', 'max']})
grouped_accidents


print(grouped_accidents)


# #### Accident severity  and  number of casulaties in different  Years

# In[27]:


df.groupby("Year")['NUMBER_OF_CASUALTIES'].sum().plot(kind='bar',title = "Accident Severity and total number of Casualties")
plt.ylabel("Total number of Casualties")


# #### * Hour of the day with the highest number of casulaties

# In[28]:



df.groupby("Hour")['NUMBER_OF_CASUALTIES'].sum().plot(kind='bar',title = "Total number of Casualties by Hours")
plt.ylabel("Total number of Casualties")


# In[29]:


#### * Count of  Traffic Accident and Vehicles involved


# In[30]:


df.groupby("NUMBER_OF_VEHICLES")['NUMBER_OF_CASUALTIES'].sum().plot(kind='bar',title = "Number of vehicles and total number of Casualties")
plt.ylabel("Total number of Casualties")


# #### * Day of the week with the highest number of casulaties

# In[31]:


grouped_df= df.groupby(['ACCIDENT_SEVERITY','DAY_OF_WEEK']).sum()
grouped_df.head(5)


# In[32]:


df.groupby(['DAY_OF_WEEK','ACCIDENT_SEVERITY']).agg({'NUMBER_OF_CASUALTIES':sum}).plot(kind='bar',figsize = (12,7))    # Sum duration


# ### Accidents categorized in Day of Week

# In[33]:


## Plotting the pie chart 
plt.figure(figsize=(45, 18))
df.groupby(['DAY_OF_WEEK']).sum().plot(kind='pie', y='NUMBER_OF_VEHICLES',autopct='%1.0f%%', fontsize=10)
plt.legend(fontsize=8, loc='upper left')
plt.show()




# ### Map distribution of Accidents across the city

# In[34]:


plt.scatter(x=df['LONGITUDE'], y=df['LATITUDE'])
plt.rcParams["figure.figsize"] = (70,45)
plt.show()


# In[35]:


import plotly.express as px
plt.figure(figsize=(15, 10))

fig = px.scatter_mapbox(df, 
                        lat='LATITUDE',
                        lon='LONGITUDE',
                        
                         center={'lat':55.860916,
                                'lon':-4.251433},
                        
                        
                    
                        zoom=13)



fig.update_layout(mapbox_style='open-street-map')


fig.show()


# In[36]:


#set seaborn plotting aesthetics
sns.set(style='white')
#sns.set(font_scale = 8)
plt.figure(figsize=(10, 6))

#create grouped bar chart
sns.barplot(x='DAY_OF_WEEK', y='NUMBER_OF_CASUALTIES', hue='ACCIDENT_SEVERITY', data=df, palette=['blue', 'orange','red'])
#add overall title
plt.title('Number of Accident by Severity & Day of Week', fontsize=20)

#add axis titles
plt.xlabel('Day of Week')
plt.ylabel('Number of Casualties')


# ### Plot number of accidents by Road Type

# In[37]:


df.groupby("DAY_OF_WEEK")['NUMBER_OF_CASUALTIES'].sum().sort_values(ascending=True).plot(kind='bar',figsize = (12,7),title = '"Days of the week and total Casualities"')
plt.ylabel("Total Casualities")


# In[38]:


#set seaborn plotting aesthetics

plt.figure(figsize=(10, 8))
sns.barplot(x="DAY_OF_WEEK", 
            y="NUMBER_OF_CASUALTIES", 
            hue="ACCIDENT_SEVERITY", 
            data=df)
plt.ylabel("Number of Casualties", size=14)
plt.xlabel("Day of Week size", size=14)
plt.title("Number of Accident by Severity & Day of Week", size=18)
plt.savefig("grouped_barplot_Seaborn_barplot_Python.png")


# ### Plot number of accidents by Speed limit

# In[39]:


plt.figure(figsize=(10, 8))
sns.barplot(x="SPEED_LIMIT", 
    y="NUMBER_OF_CASUALTIES",
    data=df,
    ci=None )
plt.ylabel("Number of Casualties", size=14)
plt.xlabel("Speed Limit", size=14)
plt.title("Number of casulties  and Speed limit", size=18)
plt.show()


# ### Plot number of accidents by Weather Condition

# In[40]:


plt.figure(figsize=(10, 8))
chart=sns.barplot(x="WEATHER_CONDITIONS", 
    y="NUMBER_OF_CASUALTIES",
    data=df,
    ci=None )
plt.ylabel("Number of Casualties", size=14)
plt.xlabel("Weather Condition", size=14)
plt.title("Number of casulties at different Weather Condition", size=18)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()


# 
# ## <a id='5'> 5. Modelling and Evaluation</a>

# ### 5.1 Encoding the categorical variables

# In[41]:


import category_encoders as ce
ce_ordinal = ce.OrdinalEncoder(cols=['ACCIDENT_SEVERITY']) # create an object of the OrdinalEncoding

df=ce_ordinal.fit_transform(df)  # fit and transform and you will get the encoded data
df.head()


# In[42]:


# cast categorical features - currently stored as string data - to their proper data format
#for col in ['ACCIDENT_SEVERITY', 'DAY_OF_WEEK', 'SPEED_LIMIT', 'URBAN_OR_RURAL_AREA',
           # 'SPECIAL_CONDITIONS_AT_SITE', 'WEATHER_CONDITIONS', 'ROAD_TYPE']:
   # df[col] = df[col].astype('category')
    
from sklearn.preprocessing import LabelEncoder
    
categ = ['DAY_OF_WEEK', 'SPEED_LIMIT', 'URBAN_OR_RURAL_AREA','SPECIAL_CONDITIONS_AT_SITE', 'WEATHER_CONDITIONS', 'ROAD_TYPE']

# Encode Categorical Columns
le = LabelEncoder()
df[categ] = df[categ].apply(le.fit_transform)   
    
df.info()


# ### 5.2 Finding the  correlation matrix

# In[43]:


# finding the  correlation matrix
# plot correlation heatmap to find out correlations
import matplotlib.pyplot as plt
import seaborn as sns

correlation =df.corr().style.background_gradient(cmap='coolwarm')
correlation


# In[45]:


# Rearrange columns

df1=df[['NUMBER_OF_VEHICLES','SPEED_LIMIT','WEATHER_CONDITIONS','NUMBER_OF_CASUALTIES', 'DAY_OF_WEEK','FIRST_ROAD_NUMBER',
'Hour','URBAN_OR_RURAL_AREA', 'SPECIAL_CONDITIONS_AT_SITE','ROAD_TYPE','ACCIDENT_SEVERITY']]

df1.head()


# In[ ]:





# # Exploartory data analysis 2

# In[46]:


# finding the  correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
correlation = df1.corr()
plt.figure(figsize = (20,20))
sns.heatmap(correlation,annot=True)


# In[47]:


# sorting the correlation in descending order
correlation['ACCIDENT_SEVERITY'].sort_values(ascending=False)


# # Method 1-Multiple linear regression and Feature importance

# In[48]:


X=df1[['NUMBER_OF_VEHICLES','SPEED_LIMIT','WEATHER_CONDITIONS','NUMBER_OF_CASUALTIES', 'DAY_OF_WEEK','FIRST_ROAD_NUMBER',
'Hour','URBAN_OR_RURAL_AREA', 'SPECIAL_CONDITIONS_AT_SITE','ROAD_TYPE','ACCIDENT_SEVERITY']]

Y=df1['ACCIDENT_SEVERITY']


# In[49]:


# with sklearn
import sklearn
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X, Y)


# In[50]:


from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# # Method 2  Feature Importance using coefficient of model

# In[51]:


# define the model
# X=df1[['NUMBER_OF_VEHICLES','SPEED_LIMIT','WEATHER_CONDITIONS','NUMBER_OF_CASUALTIES', 'DAY_OF_WEEK','FIRST_ROAD_NUMBER',
#'Hour','URBAN_OR_RURAL_AREA', 'SPECIAL_CONDITIONS_AT_SITE','ROAD_TYPE',]]
#Y=df1[['ACCIDENT_SEVERITY']]

X = df1.iloc[:, :-1]
Y = df1.iloc[:, -1]

#model = LinearRegression()
model=linear_model.LinearRegression()
# fit the model
model.fit(X, Y)

# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.title("Bar Chart of MLR Coefficients as Feature Importance Scores", size=78)
plt.xlabel("Feature", size=60)
plt.ylabel("Contribution to Model Prediction", size=60)
plt.show()


# ## Using Logistic Regression and Generating Level of Confidence

# In[52]:


from sklearn.linear_model import LogisticRegression

X = df1.iloc[:,:-1].values #price is the last columns. this line means handle all the columns except the last one
y = df1.iloc[:,-1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[53]:


pipeline_clf =  LogisticRegression(C=10, fit_intercept=True, intercept_scaling=1,n_jobs=None,
                                    penalty='l2', random_state=42,
                                    solver='newton-cg')

pipeline_clf.fit(X_train, y_train)


# In[54]:


# get predicted values from test set
y_pred = pipeline_clf.predict(X_test)

# create dataframe for output

# get confidence level of predictions
probas = np.max(pipeline_clf.predict_proba(X_test), axis=1)

# create dataframe
output = pd.DataFrame(X_test)
output['Prediction'] = y_pred
output['Confidence'] = probas
output = output.reset_index(drop=True)

print('Output:')
display(output)


# ###  Model Evaluation of Model- Method 1

# In[55]:


df1.head(2)


# In[63]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = df1.iloc[:,:-1].values #price is the last columns. this line means handle all the columns except the last one
y = df1.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)
linReg=LinearRegression()
linReg.fit(x_train,y_train)


# In[57]:



y_pred = linReg.predict(x_test)

## using  different metrics
test_set_rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
test_set_r2 = r2_score(y_test, y_pred)

print("RMSE is :",test_set_rmse) # Note that for rmse, the lower that value is, the better the fit
print("The r2 is:",test_set_r2)  # The closer towards 1, the better the fit


# In[58]:


#df1.to_csv("road_safety.csv")   #saving the current df1- propably for further use


# ###  Model Evaluation of Model Method 2

# In[59]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x=df1[['NUMBER_OF_VEHICLES','SPEED_LIMIT','WEATHER_CONDITIONS','NUMBER_OF_CASUALTIES', 'DAY_OF_WEEK','FIRST_ROAD_NUMBER',
'Hour','URBAN_OR_RURAL_AREA', 'SPECIAL_CONDITIONS_AT_SITE','ROAD_TYPE']]

y=df1['ACCIDENT_SEVERITY']


# In[60]:


#Model
regressor=LinearRegression()
regressor.fit(x,y)


# In[70]:


#Accuracy
predictions= regressor.predict(x)

mae =0
for i in range (0,len(predictions)):
    prediction =predictions[i]
    actual =y.iloc[i]
    error=abs(actual -prediction)
    mae=mae+error
    
    mae=mae/len(predictions)


# In[62]:


mae


# # Feature Importance

# ####  Feature selection, the process of finding and selecting the most useful features in a dataset, is a crucial step of the machine learning pipeline. Unnecessary features decrease training speed, decrease model interpretability, and, most importantly, decrease generalization performance on the test set.

# ## Feature Importance </b>
# *  This  is the techniques that assign a score to input features based on how useful they are at predicting a target variable</b>
# * There are many types and sources of feature importance scores, although popular examples include statistical correlation scores, coefficients calculated as part of linear models, decision trees, and permutation importance scores</b>
# * Feature importance scores play an important role in a predictive modeling project, including providing insight into the data, insight into the model </b>
# * Feature importance refers to a class of techniques for assigning scores to input features to a predictive model that indicates the relative importance of each feature when making a prediction </b>
# * Feature importance scores can be calculated for problems that involve predicting a numerical value, called regression, and those problems that involve predicting a class label, called classification.

# In[71]:


# Define the columns for X and y
X = df1.iloc[:,:-1].values #price is the last columns. this line means handle all the columns except the last one
y = df1.iloc[:,-1].values


# Split the  X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)


# In[72]:


# Checking the Rows and Columns of the X and y
X.shape, y.shape


# In[74]:


print(X_train.shape)
print(X_test.shape)


# In[75]:


lr=LogisticRegression(solver="liblinear")
lr.fit(X, y)


# In[76]:


col=['NUMBER_OF_VEHICLES','SPEED_LIMIT','WEATHER_CONDITIONS','NUMBER_OF_CASUALTIES', 'DAY_OF_WEEK','FIRST_ROAD_NUMBER',
'Hour','URBAN_OR_RURAL_AREA', 'SPECIAL_CONDITIONS_AT_SITE','ROAD_TYPE']


# In[81]:


importance = lr.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.title("Bar Chart of MLR Coefficients as Feature Importance Scores", size=78)
plt.xlabel("Feature", size=60)
plt.ylabel("Contribution to Model Prediction", size=60)
plt.xticks(rotation=90)
plt.show()


# ## Feature imporatnce after Standarization

# In[82]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[83]:


lr=LogisticRegression(solver="liblinear")
lr.fit(X, y)


# In[84]:


importance = lr.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.figure(figsize=(10,10))
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# * Standarization affect the features importance of the dataset </b>
# * As in above graph we can see that there wwre few features whose values were either too negative or positive. </b>
# * After standarization we can see that a lot of the features wholse coefficeint values are much better.</b>
# * Here negative values indicate that it tries to push the model towards the negative side </b>
# * Same case with the positive value which tends to push the model in positive side.</b>

# https://www.kaggle.com/code/prashant111/random-forest-classifier-feature-importance/notebook 

# In[98]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)
X_train.shape


# In[102]:


model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test) 

mlAcc = accuracy_score(y_test, predictions)
print('The Accuracy of the model is : %s' % mlAcc)
print('\n')
print(classification_report(y_test, predictions))

importances = pd.DataFrame(df ={'Attribute': X_train.columns, 'Importance':model.coef_[0]})
importances = importances.sort_values(by='Importance', ascending=False)


# In[93]:


X_train.columns


# In[103]:


importances


# In[ ]:




