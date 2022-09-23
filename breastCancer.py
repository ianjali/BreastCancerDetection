
#%%

#importing required libraries
import pandas as pd #for datatset 
import numpy as np
import matplotlib.pyplot as plt #for visualization
from sklearn.model_selection import train_test_split 

#classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#checking how well the classifier is working, precision and recall
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import pickle
import plotly.graph_objects as graphObj
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, precision_recall_curve, auc
from plotly.subplots import make_subplots
import itertools
from plotly.offline import plot, iplot, init_notebook_mode
#%%
#reading file 
#having different features of the nucleus of the cell
df = pd.read_csv('data.csv')
df.head() # (5*33) 
df.info()
#checking null values
df.isna().sum()  
#Unnamed: 32 : 569
#0 id| 32  Unnamed: 32 is unnecessary
#drop them  
df_clean = df.drop(columns='id',axis = 1)
df_clean = df_clean.drop(columns='Unnamed: 32',axis = 1)
#%%
#Data Visualization
#Box Plot for outliers detection from smoothness_mean to fractal_dimension_mean
names = df_clean.columns[5:12]
# convert DataFrame to list
values=[] 
for column in df_clean.iloc[:,5:12].columns:
    li = df_clean[column].tolist()
    values.append(li)
colors = ['lightblue', 'yellow', 'darkorange', 'lightgreen','cyan', 'royalblue']

fig = graphObj.Figure()
for xd, yd, cls in zip(names, values, colors):
        fig.add_trace(graphObj.Box(
            y=yd,
            name=xd,
            boxpoints='outliers',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker_size=3,
            line_width=2)
        )
fig.show()

#Box Plot for outliers detection :from texture_se to concavity_se
names = df_clean.columns[12:19]
# convert DataFrame to list
values=[] 
for column in df_clean.iloc[:,5:19].columns:
    li = df_clean[column].tolist()
    values.append(li)
colors = ['lightblue', 'yellow', 'darkorange', 'lightgreen','cyan', 'royalblue']

fig = graphObj.Figure()
for xd, yd, cls in zip(names, values, colors):
        fig.add_trace(graphObj.Box(
            y=yd,
            name=xd,
            boxpoints='outliers',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker_size=3,
            line_width=2)
        )
fig.show()
#feature distribution
sns.pairplot(df_clean.iloc[:,:8],hue='diagnosis', diag_kind='hist',height=2.0)

df_clean.info()

#correlation matrix
corr = df.iloc[:,1:].corr()
fig = graphObj.Figure(data=graphObj.Heatmap(z=np.array(corr),x=corr.columns.tolist(),y=corr.columns.tolist(),xgap = 1,ygap = 1))
fig.update_layout(margin = dict(t=25,r=0,b=200,l=200),width = 1000, height = 700)
fig.show()
#%%
#preprocessing data
#labelencoding
df_clean['diagnosis'].unique()
df_clean['diagnosis'] = df_clean['diagnosis'].map({'M' : 1, 'B': 0}) 
df_clean['diagnosis'].value_counts()
# %%
df_clean.columns
# %%
#data split
random_state = 42
X_train, X_test, y_train, y_test= train_test_split(df_clean.iloc[:,1:], df_clean['diagnosis'], test_size = 0.2, random_state = random_state)
# %%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(df_clean.shape)
#%%

#scaling using robustscaler- since there were outliers in boxplot 
scale = RobustScaler()
train_x = scale.fit_transform(X_train)
test_x = scale.transform(X_test)
# %%
pca = PCA()
pca.fit(train_x)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

fig = px.line(x=np.arange(1,exp_var_cumul.shape[0]+1), y=exp_var_cumul, markers=True, labels={'x':'# of components', 'y':'Cumulative Explained Variance'})

fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=30, y0=0.95, y1=0.95)
fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=30, y0=1, y1=1)

fig.show()

#%%
# Fit the 
# Fit the RFE model to identify the optimum number of features .
rfecv = RFECV(cv=StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True),
      estimator=DecisionTreeClassifier(), scoring='accuracy')
    
rfecv.fit(train_x, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(15,5))
plt.xlabel("# of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(
    range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
    rfecv.cv_results_['mean_test_score'],
)
plt.show()

#%%
# Identifying the features RFE selected
df_features = pd.DataFrame(columns = ['feature', 'support', 'ranking'])

for i in range(train_x.shape[1]):
    row = {'feature': i, 'support': rfecv.support_[i], 'ranking': rfecv.ranking_[i]}
    df_features = df_features.append(row, ignore_index=True)
    
df_features.sort_values(by='ranking').head(10)

# %%
#finding best hyperparameters
def modelselection(classifier, parameters, scoring, X_train):
    clf = GridSearchCV(estimator=classifier,
                   param_grid=parameters,
                   scoring= scoring,
                   cv=5,
                   n_jobs=-1)# n_jobs refers to the number of CPU's that you want to use for excution, -1 means that use all available computing power.
    clf.fit(X_train, y_train)
    cv_results = clf.cv_results_
    best_parameters = clf.best_params_
    best_result = clf.best_score_
    print('The best parameters for classifier is', best_parameters)
    print('The best training score is %.3f:'% best_result)
#    print(sorted(cv_results.keys()))
    return cv_results, best_parameters, best_result
# of Components in PCA versus Model Accuracy/Training Time
def PCA_curves(PCA_cv_score, PCA_test_score, training_time):
    fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    subplot_titles=('# of Components in PCA versus Model Accuracy','# of Components in PCA versus Training Time')
    )
    
    fig.add_trace(graphObj.Scatter(x=n,y=PCA_cv_score,
                             line=dict(color='rgb(231,107,243)', width=2), name='CV score'),
                  row=1, col=1)
    fig.add_trace(graphObj.Scatter(x=n,y=PCA_test_score,
                             line=dict(color='rgb(0,176,246)', width=2), name='Test score'),              
                  row=1, col=1)    
    fig.add_trace(graphObj.Scatter(x=n,y=training_time,
                             line=dict(color='rgb(0,100,80)', width=2), name='Training time'),
                  row=1, col=2)
    fig.update_xaxes(title_text='# of components')
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    # fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text='Training time', row=1, col=2)
    fig.show()
#%%
def metrics(X,CV_clf):
    y_pred = CV_clf.predict(X)
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    tp = cm[1,1]
    Accuracy=(tp+tn)/(tp+tn+fp+fn)
    Sensitivity=tp/(tp+fn)
    Specificity=tn/(tn+fp)
    Precision=tp/(tp+fp)
    F_measure=2*tp/(2*tp+fp+fn)
    print('Accuracy=%.3f'%Accuracy)
    print('Sensitivity=%.3f'%Sensitivity) # as the same as recall
    print('Specificity=%.3f'%Specificity)
    print('Precision=%.3f'%Precision)
    print('F-measure=%.3f'%F_measure)
    return Accuracy, Sensitivity, Specificity, Precision, F_measure
 #   plot_confusion_matrix(CV_clf, X_test, y_test)
 
def plot_roc_prc():
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=(f'ROC Curve (AUC={auc(fpr, tpr):.4f})',f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})')
    )
    fig.add_trace(graphObj.Scatter(x=fpr, y=tpr),row=1, col=1)
    fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1,row=1, col=1)
    fig.add_trace(graphObj.Scatter(x=recall, y=precision),row=1, col=2)
    fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0.5, y1=0.5,row=1, col=2)
    # Update axis properties
    fig.update_xaxes(title_text="False Positive Rate / 1-Specificity", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate / Recall", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.show()
#%%
classifier_log = LogisticRegression(random_state=random_state,solver='lbfgs', max_iter=1000)
parameters_log = {
            'penalty' : ['l2'],  
            'C' : [0.01, 0.1, 1, 10, 100]
}
scoring='accuracy'   # scoring parameters: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# Find the best hyperparameters
cv_results, best_param, best_result = modelselection(classifier_log,parameters_log, scoring, train_x)
# %%

# Classifier with the best hyperparameters
logReg_clf = LogisticRegression(penalty = best_param['penalty'],
                            C = best_param['C'],
                            random_state=random_state)
logReg_clf.fit(train_x, y_train)

# Metrics
logReg_metrics = metrics(test_x,logReg_clf)
#%%
#logistic regression with PCA
def compare_pca(n_components):
    cv_score, test_score, cv_training_time = [], [], []
    for n in n_components:
        print("The number of components in PCA is:%d "% n)
        pca = PCA(n_components=n, svd_solver="full",random_state=random_state)
        X_PCA_train = pca.fit_transform(train_x)
        X_PCA_test = pca.transform(test_x)
        # Model Selection
        cv_results, best_param, best_result = modelselection(classifier_log,parameters_log, scoring, X_PCA_train)
        training_time = np.mean(np.array(cv_results['mean_fit_time'])+np.array(cv_results['mean_score_time']))
        cv_score.append(best_result)
        cv_training_time.append(training_time)
        CV_clf = LogisticRegression(penalty = best_param['penalty'],
                                    C = best_param['C'],
                                    random_state=random_state)
        CV_clf.fit(X_PCA_train, y_train)
        score = CV_clf.score(X_PCA_test, y_test)
        test_score.append(score)
    print(cv_score, test_score, cv_training_time)
    return cv_score, test_score, cv_training_time
# %%
n_features = train_x.shape[1]
n = np.arange(2, n_features+2, 2) 

PCA_cv_score, PCA_test_score, PCA_cv_training_time= compare_pca(n_components = n)
#%%
PCA_curves(PCA_cv_score,PCA_test_score,PCA_cv_training_time)
#%%
i =PCA_test_score.index(max(PCA_test_score))
print('The best accuracy of logistic regression classifier is: %.3f'%  max(PCA_test_score)+', where the total number of components in PCA is {:.0f}'.format((i+1)*2))
#%%
pca = PCA(n_components=(i+1)*2, svd_solver="full",random_state=random_state)
X_PCA_train = pca.fit_transform(train_x)
X_PCA_test = pca.transform(test_x)
# Model Selection
cv_results, best_param, best_result = modelselection(classifier_log,parameters_log, scoring, X_PCA_train)

# Classifier with the best hyperparameters
logReg_PCA = LogisticRegression(penalty = best_param['penalty'],
                            C = best_param['C'],
                            random_state=random_state)
logReg_PCA.fit(X_PCA_train, y_train)

# Metrics
logReg_PCA_metrics = metrics(X_PCA_test,logReg_PCA)

# ROC Curve & Precision-Recall Curves
y_score = logReg_PCA.predict_proba(X_PCA_test)[:, 1] # predict probabilities
plot_roc_prc()
#%%
thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

for n, ax in zip(thresholds,axs.ravel()):
    y_score = logReg_PCA.predict_proba(X_PCA_test)[:,1] > n
    
    cm = confusion_matrix(y_test, y_score)
    
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]

    print('threshold = %s :'%n,
          'Accuracy={:.3f}'.format((tp+tn)/(tp+tn+fp+fn)),
          'Sensitivity={:.3f}'.format(tp/(tp+fn)),
          'Specificity={:.3f}'.format(tn/(tn+fp)),
          'Precision={:.3f}'.format(tp/(tp+fp)))
    
    im=ax.matshow(cm, cmap='Blues', alpha=0.7)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        ax.text(j, i, cm[i, j], horizontalalignment = 'center')
        
    ax.set_ylabel('True label',fontsize=12)
    ax.set_xlabel('Predicted label',fontsize=12)
    ax.set_title('Threshold = %s'%n, fontsize=12)
    fig.colorbar(im, ax=ax,orientation='vertical');
plt.show()
#%%
models_metrics = {'logisticRegression': [round(elem, 3) for elem in logReg_metrics], 
                 'logReg+PCA': [round(elem, 3) for elem in logReg_PCA_metrics]
                }
index=['Accuracy','Sensitivity','Specificity','Precision', 'F-measure']
df_scores = pd.DataFrame(data = models_metrics, index=index)
ax = df_scores.plot(kind='bar', figsize = (15,6), ylim = (0.90, 1.02), 
                    color = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen','cyan'],
                    rot = 0, title ='Models performance (test scores)',
                    edgecolor = 'grey', alpha = 0.5)
ax.legend(loc='upper center', ncol=5, title="models")
for container in ax.containers:
    ax.bar_label(container)
plt.show()

#%%
#=================

# %%

#saving our model
#filename = 'model.pkl'
#pickle.dump(model, open(filename, 'wb'))
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))