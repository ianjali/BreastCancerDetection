#%%
#importing required libraries
import pandas as pd #for datatset 
import matplotlib.pyplot as plt #for visualization
from sklearn.model_selection import train_test_split 
#classifier
from sklearn.ensemble import RandomForestClassifier
#checking how well the classifier is working, precision and recall
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
#%%
#reading file 
#having different features of the nucleus of the cell
df = pd.read_csv('data.csv')
df.head() # (5*33) 
#%%
df.info()
#checking null values
df.isna().sum()  
#Unnamed: 32 : 569
#0 id| 32  Unnamed: 32 is unnecessary
#drop them  
df_clean = df.drop(columns='id',axis = 1)
df_clean = df_clean.drop(columns='Unnamed: 32',axis = 1)
# %%
df_clean.info()
# %%
df_clean['diagnosis'].unique()
df_clean['diagnosis'] = df_clean['diagnosis'].map({'M' : 1, 'B': 0}) 
df_clean.head()
# %%
df.columns
# %%
mean_features = list(df_clean.columns[1:11])
se_features = list(df_clean.columns[11:21])
worst_features = list(df_clean.columns[21:31])
# %%
mean_features.append('diagnosis')
se_features.append('diagnosis')
worst_features.append('diagnosis')

# %%
#checking correlation with diagnosis 
corr = df_clean[mean_features].corr()
corr
#%%[markdown]
#
# ## Training 

# %%
#after checking correlation values with diagnosis,
#selecting the highest correlated features 
key_vars = ['radius_mean','perimeter_mean','area_mean', 'compactness_mean','concavity_mean',
            'concave points_mean','radius_se','area_se','radius_worst','perimeter_worst','compactness_worst']

# %%
train, test = train_test_split(df_clean,test_size = 0.15, random_state=1)
train_x = train[key_vars]
train_y  = train['diagnosis']
test_x = test[key_vars]
test_y  = test['diagnosis']

# %%
model = RandomForestClassifier()
model.fit(train_x, train_y)
# %%
pred_y=model.predict(test_x)

# %%
confusion_matrix(test_y,pred_y)
# %%
#tp/(tp+fp)
print(f'Precision score for RandomClassfier is :  {precision_score(test_y,pred_y)}')
print(f'Recall score for RandomClassfier is :  {recall_score(test_y,pred_y)}')
print(f'Accuracy is : {accuracy_score(test_y,pred_y)}')
#tp/(tp+fn)
# %%

