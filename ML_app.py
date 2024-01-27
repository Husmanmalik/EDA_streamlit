#import libararies
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score

#title of the app
st.title('''
          Explore different ML model on datasets
         ''')
dataset_name =st.sidebar.selectbox(
    'Choose a dataset',
    ('Iris','Wine','Breast Cancer')    
)
classifier_name=st.sidebar.selectbox(
    'Select classifier',
    ('KNN','SVM','Random Forest')
)
# Checking which data is uploaded
def get_dataset(dataset_name):
    data=None
    if dataset_name =='Iris':
        data=datasets.load_iris()
    elif datasets=='Wine':
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x=data.data
    y=data.target
    return x,y


X,y =get_dataset(dataset_name)

st.write('Shape of dataset:',X.shape)
st.write('Number of classes:',len(np.unique(y)))

# adding different parameter of mechine learning model such as k,neighbour,max_depth
def add_parameter_ui(classifier_name):
    params=dict() # considering params is initialy empty
    if  classifier_name=='SVM':
        c=st.sidebar.slider('C',0.01,10.0)
        params['C']=c
    elif classifier_name=='KNN':
        k=st.sidebar.slider('Nearst_Neighbour',1,15)
        params['k']=k
    else:
        max_depth= st.sidebar.slider('max_depth',2,15)
        params['max_depth']=max_depth # depth of every tree that grow in random forest
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']=n_estimators
    return params
# ab hum is function ko bula lay gay base on classifier_name and params
params=add_parameter_ui(classifier_name)
# ab hum classifier bhanay gay base on classifier_name and parameter
def get_classifier(classifier_name,params):
    clf=None
    if classifier_name=='SVM':
        clf=SVC(C=params['C'])
    elif classifier_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['k'])
    else:# Random Forest Classifier
        clf=RandomForestClassifier(max_depth=params['max_depth'],n_estimators=params['n_estimators'],random_state=1234)
    return clf 
#now calling it
clf=get_classifier(classifier_name,params)
# now we will train our model using X and Y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
# model ka accuracy score check karnay ka liya
accuracy= accuracy_score(y_test,y_pred)
st.write(f'Classifier={classifier_name}')
st.write(f'Accuracy Score = {accuracy}')
# Now let us visualize the data
### PLOT DATASET ###
pca=PCA(2)
X_projected=pca.fit_transform(X)
# ab hum dimenstion may slice kar rahay ha 
X1=X_projected[:,0]
X2=X_projected[:,1]

fig=plt.figure()
plt.scatter(X1,X2,c=y,cmap='viridis')

plt.xlabel('Pricipal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)

