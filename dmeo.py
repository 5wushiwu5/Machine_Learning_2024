import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from tensorflow.python.keras import Input
from keras._tf_keras.keras.layers import Dense
from keras import Sequential

##from tensorflow.models import Sequential


plt.style.use('ggplot')
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

data = pd.read_csv('kaggle\input\web-phishing-dataset\web-page-phishing.csv')
data.head()

data.describe()

fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(20,10))
for i,col in enumerate(['url_length','n_dots']):
    ax[i].boxplot(np.log(data.loc[data['phishing'] == 0,col]), 
            vert=False, positions=[1], patch_artist=True,
            boxprops=dict(facecolor='lightgreen'), 
            medianprops=dict(color='darkgreen'))  
    ax[i].boxplot(np.log(data.loc[data['phishing'] == 1,col]), 
            vert=False, positions=[2], patch_artist=True,
            boxprops=dict(facecolor='coral'), 
            medianprops=dict(color='red'))  
    ax[i].set_title(f'Distribution of Log {col.title()}: Normal vs. Phishing')
    ax[i].set_xlabel(f'Log {col.title()}')
    ax[i].set_yticks([1, 2], ['Normal', 'Phishing'])
    
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,10))
plt.boxplot((data.loc[data['phishing'] == 0,'n_redirection']), 
            vert=False, positions=[1], patch_artist=True,
            boxprops=dict(facecolor='lightgreen'), 
            medianprops=dict(color='darkgreen'))  
plt.boxplot((data.loc[data['phishing'] == 1,'n_redirection']), 
            vert=False, positions=[2], patch_artist=True,
            boxprops=dict(facecolor='coral'), 
            medianprops=dict(color='red'))  
plt.title('Distribution of Log Number of Redirections: Normal vs. Phishing')
plt.xlabel('Log URL Length')
plt.yticks([1, 2], ['Normal', 'Phishing'])
plt.tight_layout()
plt.show()

descriptive_stats = data.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]].groupby('phishing').describe()

stat_data = descriptive_stats.xs('mean', level=1, axis=1)
stat_data = np.sqrt(stat_data)

n_features = len(stat_data.columns)
index = np.arange(n_features)
bar_width = 0.35

plt.figure(figsize=(14, 8))
plt.bar(index, stat_data.iloc[0], bar_width, label='Non-Phishing')
plt.bar(index + bar_width, stat_data.iloc[1], bar_width, label='Phishing')

plt.xlabel('Features')
plt.ylabel('Mean Value')
plt.title('Mean Values of Features by Phishing Category')
plt.xticks(index + bar_width / 2, stat_data.columns, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),cmap='coolwarm',annot=True,cbar=False)
plt.show()

# def logistic_regression_summary(model, features):
#     coefficients = np.concatenate([model.intercept_, model.coef_.flatten()])
#     pred_probs = model.predict_proba(features)
#     X_design = np.hstack([np.ones((features.shape[0], 1)), features])
#     V = np.diagflat(np.prod(pred_probs, axis=1))
#     cov_matrix = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))
#     standard_errors = np.sqrt(np.diag(cov_matrix))
#     z_values = coefficients / standard_errors
#     p_values = [2 * (1 - stats.norm.cdf(np.abs(z))) for z in z_values]
#     conf_intervals = [stats.norm.ppf(0.975) * se for se in standard_errors]
#     report = {
#         'coef': coefficients,
#         'std err': standard_errors,
#         'z': z_values,
#         'P>|z|': p_values,
#         '[0.025': coefficients - conf_intervals,
#         '0.975]': coefficients + conf_intervals,
#     }
    
#     return report


X = data.drop(['phishing'],axis=1).values
y = data['phishing'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
# report = logistic_regression_summary(model, X_train)

# feature_names = data.columns.tolist()[:-1]
# report_df = pd.DataFrame(report, index=['intercept'] + feature_names) 
# print(report_df)

def confusion_matrix_plot(y_test,predictions):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", 
                cmap="Blues",
                square=True,
                cbar=False,
                xticklabels=['False', 'True'],
                yticklabels=['False', 'True'])
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(2) + 0.5
    plt.xticks(tick_marks, ['Non-phishing', 'Phishing'], rotation=0)
    plt.yticks(tick_marks, ['Non-phishing', 'Phishing'], rotation=0)
    plt.tight_layout()
    plt.show()

def area_under_curve(model,X_test,y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.show()




def calculate_metrics(model,X_test,y_test):
    predictions = model.predict(X_test)
    print(f"Accuracy Score: {accuracy_score(y_test,predictions)}")
    confusion_matrix_plot(y_test,predictions)
    area_under_curve(model,X_test,y_test)
    
calculate_metrics(model,X_test,y_test)

##随机森林#########################################################################################################################
# model = RandomForestClassifier(n_estimators=80,max_depth=18,max_features='sqrt',min_samples_split=12,criterion='gini')
# model.fit(X_train,y_train)

# predictions_train = model.predict(X_train)
# predictions_test = model.predict(X_test)

# print(f"Train Accuracy Score: {accuracy_score(y_train,predictions_train)}")
# print(f"Test Accuracy Score: {accuracy_score(y_test,predictions_test)}")

##################################################################################################################################


##神经网络
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model = Sequential([
#     Input(shape=(X_train.shape[1],)), 
#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid') 
# ])

# model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# history = model.fit(X_train, y_train, epochs=25,validation_split=0.2,)
##