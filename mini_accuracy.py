import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from fcmeans import FCM
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from kneed import KneeLocator
import math

def readData(s):
    data = pd.read_excel(s)
    return data

def labelEncoder(data, category):
    global d
    if d.get(category,None)==None:
        d[category] = {'dictionary':dict(), 'count':0}
    res = []
    for val in data[category]:
        if d[category]['dictionary'].get(val,None)==None:
            d[category]['dictionary'][val] = d[category]['count']
            d[category]['count']+=1
        res.append(d[category]['dictionary'][val])
  # label_encoder = preprocessing.LabelEncoder()
    data[category] = res
    return data

def getXY(data, category):
    x = data.drop([category],axis = 1)
    y = data[category]
    return x,y

def borutaGreenBlue(x,y):
    forest = RandomForestRegressor(n_jobs = -1, max_depth = 5)
    boruta = BorutaPy(estimator = forest, n_estimators = 'auto', max_iter = 100)
    boruta.fit(np.array(x), np.array(y))

    green_area = x.columns[boruta.support_].to_list()
    blue_area = x.columns[boruta.support_weak_].to_list()
    return 'features in the green area:' + str(green_area) + '\nfeatures in the blue area:' + str(blue_area)

def doFCM(x,numOfClusters):
    numpy_array = x.to_numpy()
    fcm = FCM(n_clusters=numOfClusters)
    fcm.fit(numpy_array) 
    res = []

    for ele in fcm.u:
        tempEle = list(ele)
        if max(tempEle)<0.65:
            res.append(numOfClusters)
        else:
            res.append(tempEle.index(max(tempEle)))
    return res

def getCluster(clusterNumber, x, y, clustersList):
    resX, resY = [],[]
    for row in range(len(x)):
        if clustersList[row]==clusterNumber:
            resX.append(x.iloc[row])
            resY.append(y.iloc[row])
  
    resX = pd.DataFrame(resX)
    return resX, resY

def predict(trainX, trainY, testX, numberOfClusters):
    n = len(testX)
    finalRes = []
    for i in range(n):
        addedX = trainX.append(testX.iloc[i])
        res = doFCM(addedX,numberOfClusters)
        #print(res)
        trainClusterX, trainClusterY = getCluster(res[-1],trainX, trainY, res[:-1])
    
        regressor = RandomForestRegressor(n_estimators = 1000, random_state = 50)
        model = regressor.fit(trainClusterX, trainClusterY)
        regressorPredY = model.predict([testX.iloc[i]])

        svm = SVC(kernel='rbf', random_state=10, gamma=0.5, C=10.0)
        model = svm.fit(trainClusterX, trainClusterY)
        svmPredY = model.predict([testX.iloc[i]])

        svp = SVC(kernel='linear')
        model = svp.fit(trainClusterX, trainClusterY)
        svpPredY = model.predict([testX.iloc[i]])

        finalRes.append((regressorPredY + svmPredY + svpPredY)/3)
    return finalRes

#main starts here

d = dict()

#read existing data
hospitalData = readData('./data.xlsx')

#reading have to predict data
testX = readData('./testData.xlsx')

#dropping unnecessary columns
hospitalData = hospitalData.drop(['ID','date admission'],axis=1)
#testX = testX.drop(['ID','date admission'],axis=1)

#labelEncoding all categorical values
categoricalColumns = ['age','sex','weight','diagnose','sub diagnose','flora','medicamant','active substance']
for category in categoricalColumns:
  hospitalData = labelEncoder(hospitalData, category)
# for category in categoricalColumns:
#   testX = labelEncoder(testX, category)

#dividing dataset into x & y
x, y = getXY(hospitalData, 'time hospital')
trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0 )

#checking for appropriate K
wcss=[]
K = range(1,10)
for i in K:
  kmeans = KMeans(i)
  kmeans.fit(x)
  wcss_iter = kmeans.inertia_
  wcss.append(wcss_iter)

kl = KneeLocator(K, wcss, curve="convex", direction="decreasing")
numberOfClusters = kl.elbow

#predicting Results
predY = predict(trainX, trainY, testX, numberOfClusters)

mse = metrics.mean_squared_error(testY, predY)
mae = metrics.mean_absolute_error(testY,predY)
r2 = metrics.r2_score(testY,predY)
print("RootMeanSquaredError {}\nMeanAbsoluteError {}\nRSquared {}".format(math.sqrt(mse),mae,r2))