# Reads in player names, positions, and ratings
# into three different arrays with consistent indexing
import sys
import random
import numpy as np
import scipy.stats as stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.cross_validation import KFold
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn import linear_model

#Determined values from the tests below
# Comparison Algorithms with and without feature selection
randomMse = 20.197512119571048
linearMseWithFullFeatureSet = 7.58423316042
ridgeRegressionMseWithFullFeaturesSet = 7.58423084482
radiusRegressorMseWithFullFeatureSet = 7.50028337203     # radius = 67
radiusFeatures=[0, 6, 7, 10, 11, 13, 16, 24, 26]

# kNN Regression model
kNNRegressSfsFeatures=[4,6,7,24,27]
kNNRegressSfsMse = 6.5981770546675502
kNNRegressSfsNeighbors = 29
kNNRegressExpertFeatures = [6,7,8,9,10,11,19,22]
kNNRegressExpertNeighbors = 33
kNNRegressExpertMse = 6.92513018612
kNNRegressSfsPlusExpertFeatures = [4,6,7,8,9,10,11,19,22,24,27]
kNNRegressSfsPlusExpertMse = 6.8936594168
kNNRegressSfsPlusExpertNeighbors = 57
kNNL2MseFullFeatureSet = 7.96849050972
kNNL1MseFullFeatureSet = 7.80139799285
knnChessMse = 8.1577929509
knnCanberraMse = 8.00405226917
knnBrycurtisMse = 7.88338225114

# Read in player names
names = []
with open("FinalResults_Names.txt") as nameFile:
  for line in nameFile :
    names.append(line.strip('\n'))

# Read in player positions
positions = []
with open("FinalResults_Positions.txt") as posFile:
  for line in posFile :
    positions.append(line.strip('\n'))

# Read in player ratings
ratingsFile = open("FinalResults_Ratings_Minus_Overall.txt")
ratings = np.loadtxt(ratingsFile, delimiter=",")
ratings_normalized = preprocessing.normalize(ratings)
ratings_standardized = preprocessing.scale(ratings)

# Read in legends
legendsFile = open("Legends_No_Goals.txt")
legends = np.loadtxt(legendsFile, delimiter=",")

# Read in legends goals
legendsGoalsFile = open("Legends_Goals.txt")
legendsGoals = np.loadtxt(legendsGoalsFile, delimiter=",")

# Read in player goal tallies
goalsFile = open("FinalResults_Goals.txt")
goals = np.loadtxt(goalsFile)

# Random goal generator
#randomSseArray = []
#for train_index, test_index in folds:
#  # Generate predictions
#  ranPredictions = []
#  for i in (range(0, len(legends))):
#    ranPredictions.append(random.choice(legendsGoals))
#  randomSseArray.append(mean_squared_error(legendsGoals, ranPredictions))
#print('Random %f' % np.asarray(randomSseArray).mean())

# Linear Regression, full features
folds = KFold(12365, n_folds=10)
linearMseArray=[]
for train_index, test_index in folds:
  linearModel = linear_model.LinearRegression()
  linearModel.fit(ratings[train_index], goals[train_index])
  linearPred = linearModel.predict(ratings[test_index])
  linearMse = mean_squared_error(goals[test_index], linearPred)
  linearMseArray.append(linearMse)
print('Linear regression %f' % np.asarray(linearMseArray).mean())

# Ridge Regression, full features
#folds = KFold(12365, n_folds=10)
#ridgeMseArray=[]
#for train_index, test_index in folds:
#  ridgeModel = linear_model.RidgeCV(alphas=[0.001,0.001,0.01,0.1,1,2,5,10,20])
#  ridgeModel.fit(ratings, goals)
#  ridgePred = ridgeModel.predict(legends)
#  ridgeMse = mean_squared_error(legendsGoals, ridgePred)
#  ridgeMseArray.append(ridgeMse)
#print('Ridge regressor %f' % np.asarray(ridgeMseArray).mean())
#linearMse = 7.5842331604168889
#ridgeMse = 7.58423084482

# Radius Regressor, full features
#radArray=[]
#for i in range(0,100):
# tree = \
# KDTree(ratings[:,[radiusFeatures]].reshape(12365, len(radiusFeatures)))
# kModel = RadiusNeighborsRegressor(algorithm="kd_tree", radius=33, metric='manhattan')
# kModel.fit(tree, goals)
# kPred = kModel.predict(legends[:,[radiusFeatures]].reshape(54, len(radiusFeatures)))
# radArray.append(mean_squared_error(legendsGoals, kPred))
#print('Radius regressor %f', np.asarray(radArray).mean())

## Feature selection first!
# Human Expert
#selectedFeatures=[6,7,8,9,10,11,19,22]
#folds = KFold(12365, n_folds=10)
#expertSseArray=[]
#for train_index, test_index in folds:
#  tree = \
#    KDTree(ratings[:,[selectedFeatures]].reshape(12365, len(selectedFeatures))[train_index])
#  expertModel = KNeighborsRegressor(algorithm="kd_tree", n_neighbors=33, metric='manhattan')
#  expertModel.fit(ratings[train_index], goals[train_index])
#  expertPred = expertModel.predict(ratings[test_index])
#  sse = mean_squared_error(goals[test_index], expertPred)
#  expertSseArray.append(sse)
#print(np.asarray(expertSseArray).mean())

## SFS
#for i in range(100, 200, 1):
#featurePool=range(0,28)
#chosenFeatures=[]
#basePerf = float("inf")
#while len(featurePool) > 0:
#  print('here')
#  bestPerf = float("inf")
#  bestFeature=-1
#  for feature in featurePool:
#    print('here@')
#    sseArray=[]
#    folds = KFold(12365, n_folds=10)
#    for train_index, test_index in folds:
#      featureSubset=ratings[:,[chosenFeatures+[feature]]].reshape(12365,len(chosenFeatures)+1)
#      model = RadiusNeighborsRegressor(algorithm="kd_tree", radius=i, metric='manhattan')
#      model.fit(featureSubset[train_index], goals[train_index])
#      prediction = model.predict(featureSubset[test_index])
#      sse = mean_squared_error(goals[test_index], prediction)
#      sseArray.append(sse)
#    mean = np.asarray(sseArray).mean()
#    if (mean < bestPerf): 
#      bestPerf = mean
#      bestFeature = feature
#  if (bestPerf < basePerf):
#    basePerf = bestPerf
#    featurePool.remove(bestFeature)
#    chosenFeatures.append(bestFeature)
#  else:
#    break
#print("Chosen features: '{0}', # of neighbors: '{1}', basePerf: '{2}'".format(chosenFeatures, i, basePerf))

# Final model run on legends database
#folds = KFold(12365, n_folds=10)
knnSseArray=[]
for train_index, test_index in folds:
  tree = \
    KDTree(ratings[:,[kNNRegressSfsFeatures]].reshape(12365, len(kNNRegressSfsFeatures))[train_index])
  kModel = KNeighborsRegressor(algorithm="kd_tree", n_neighbors=29, metric='manhattan', weights='uniform')
  kModel.fit(tree, goals[train_index])
  kPred = kModel.predict(ratings[:,[kNNRegressSfsFeatures]].reshape(12365, len(kNNRegressSfsFeatures))[test_index])
  knnSseArray.append(mean_squared_error(goals[test_index], kPred))
print('kNN Model %f' % np.asarray(knnSseArray).mean())

# Features from SFS + human expert
#comboFeatures = list(set(kNNRegressSfsFeatures + kNNRegressExpertFeatures))
#comboSseArray=[]
#for train_index, test_index in folds:
#  tree = \
#    KDTree(ratings[:,[comboFeatures]].reshape(12365, len(comboFeatures))[train_index])
#  expertModel = KNeighborsRegressor(algorithm="kd_tree", n_neighbors=57)
#  expertModel.fit(ratings[:,[comboFeatures]].reshape(12365, len(comboFeatures))[train_index], goals[train_index])
#  expertPred = expertModel.predict(ratings[:,[comboFeatures]].reshape(12365, len(comboFeatures))[test_index])
#  sse = mean_squared_error(goals[test_index], expertPred)
#  comboSseArray.append(sse)
#print(np.asarray(comboSseArray).mean())

# SBS
#for i in (95, 105, 5):
#  chosenFeatures=range(0,28)
#  basePerf = float("inf")
#  while len(chosenFeatures) > 1:
#    bestPerf = float("inf")
#    bestFeature=-1
#    for feature in chosenFeatures:
#      testFeatureSet=chosenFeatures[:]
#      testFeatureSet.remove(feature)
#      sseArray=[]
#      folds = KFold(12365, n_folds=10)
#      for train_index, test_index in folds:
#        featureSubset=ratings[:,[testFeatureSet]].reshape(12365,len(testFeatureSet))
#        tree = KDTree(featureSubset[train_index])
#        model = KNeighborsRegressor(algorithm="kd_tree", n_neighbors=i)
#        model.fit(tree, goals[train_index])
#        prediction = model.predict(featureSubset[test_index])
#        sse = mean_squared_error(goals[test_index], prediction)
#        sseArray.append(sse)
#      mean = np.asarray(sseArray).mean()
#      if (mean < bestPerf): 
#        bestPerf = mean
#        bestFeature = feature
#    if (bestPerf < basePerf):
#      basePerf = bestPerf
#      chosenFeatures.remove(bestFeature)
#    else:
#      break
#  print("Chosen features: '{0}', # of neighbors: '{1}', basePerf: '{2}'".format(chosenFeatures, i, basePerf))

# Distance tests, 10-fold cross validation
#euclideanSseArray = []
manhattanSseArray = []
#chessSseArray = []
#canberraSseArray = []
#brycurtisSseArray = []
#l2AveRatio = 0
#l1AveRatio = 0
#chessAveRatio = 0
#canberraAveRatio = 0
#brycurtisAveRatio = 0
#folds = KFold(12365, n_folds=10)
#for train_index, test_index in folds:
#  # Create k-NN Regressor
#  l2Model = KNeighborsRegressor(algorithm="ball_tree") 
#  l1Model = KNeighborsRegressor(algorithm="ball_tree", metric='manhattan') 
#  chessModel = KNeighborsRegressor(algorithm="ball_tree", metric='chebyshev')
#  canberraModel = KNeighborsRegressor(algorithm="ball_tree", metric='canberra')
#  brycurtisModel = KNeighborsRegressor(algorithm="ball_tree", metric='braycurtis')
#
#  # Fit models
#  l2Model.fit(ratings[train_index], goals[train_index])
#  l1Model.fit(ratings[:,[kNNRegressSfsFeatures]].reshape(12365, len(kNNRegressSfsFeatures))[train_index], goals[train_index])
#    KDTree(ratings[:,[kNNRegressSfsFeatures]].reshape(12365, len(kNNRegressSfsFeatures)))
#  chessModel.fit(ratings[train_index], goals[train_index])
#  canberraModel.fit(ratings[train_index], goals[train_index])
#  brycurtisModel.fit(ratings[train_index], goals[train_index])
#
#  # Predict on test set
#  l2Pred = l2Model.predict(ratings[test_index])
#  l1Pred = l1Model.predict(ratings[:,[kNNRegressSfsFeatures]].reshape(12365, len(kNNRegressSfsFeatures))[test_index])
#  chessPred = chessModel.predict(ratings[test_index])
#  canberraPred = canberraModel.predict(ratings[test_index])
#  brycurtisPred = brycurtisModel.predict(ratings[test_index])
#
#  # Get nearest neighbors
#  l2Neigh = l2Model.kneighbors(ratings[test_index], len(test_index))
#  l1Neigh = l1Model.kneighbors(ratings[:,[kNNRegressSfsFeatures]].reshape(12365, len(kNNRegressSfsFeatures))[test_index], len(test_index))
#  chessNeigh = chessModel.kneighbors(ratings[test_index], len(test_index))
#  canberraNeigh = canberraModel.kneighbors(ratings[test_index], len(test_index))
#  brycurtisNeigh = brycurtisModel.kneighbors(ratings[test_index], len(test_index))
#
#  # Calculate ratio of ave. furthest point / ave. closest point
#  l2AveRatio += (l2Neigh[0].mean(axis=0)[len(test_index) - 1] / l2Neigh[0].mean(axis=0)[0])
#  l1AveRatio += (l1Neigh[0].mean(axis=0)[len(test_index) - 1] / l1Neigh[0].mean(axis=0)[0])
#  chessAveRatio += \
#    (chessNeigh[0].mean(axis=0)[len(test_index) - 1] / chessNeigh[0].mean(axis=0)[0])
#  canberraAveRatio += \
#    (canberraNeigh[0].mean(axis=0)[len(test_index) - 1] / canberraNeigh[0].mean(axis=0)[0])
#  brycurtisAveRatio += \
#    (brycurtisNeigh[0].mean(axis=0)[len(test_index) - 1] / brycurtisNeigh[0].mean(axis=0)[0])
#
#  # Calculate sum squared error
#  l2Sse = mean_squared_error(goals[test_index], l2Pred)
#  l1Sse = mean_squared_error(goals[test_index], l1Pred)
#  chessSse = mean_squared_error(goals[test_index], chessPred)
#  canberraSse = mean_squared_error(goals[test_index], canberraPred)
#  brycurtisSse = mean_squared_error(goals[test_index], brycurtisPred)
#
#  # Add to array
#  euclideanSseArray.append(l2Sse)
#  manhattanSseArray.append(l1Sse)
#  chessSseArray.append(chessSse)
#  canberraSseArray.append(canberraSse)
#  brycurtisSseArray.append(brycurtisSse)
#
#sys.stdout.write('The Euclidean MSE = %s\n' %  np.asarray(euclideanSseArray).mean())
sys.stdout.write('The Manhattan MSE = %s\n' % np.asarray(manhattanSseArray).mean())
#sys.stdout.write('The Chess MSE = %s\n' % np.asarray(chessSseArray).mean())
#sys.stdout.write('The Canberra MSE = %s\n' % np.asarray(canberraSseArray).mean())
#sys.stdout.write('The Brycurtis MSE = %s\n' % np.asarray(brycurtisSseArray).mean())


#fValue, pValue = stats.f_oneway(euclideanSseArray, manhattanSseArray, fractionalSseArray)
#sys.stdout.write('f value = %s\n' % fValue)
#sys.stdout.write('p value = %s\n' % pValue)

#sys.stdout.write('Euclidean ratio: %f\n' % (l2AveRatio / 10))
#sys.stdout.write('Manhanttan ratio: %f\n' % (l1AveRatio / 10))
#sys.stdout.write('Chess ratio: %f\n' % (chessAveRatio / 10))
#sys.stdout.write('Canberra ratio: %f\n' % (canberraAveRatio / 10))
#sys.stdout.write('Brycurtis ratio: %f\n' % (brycurtisAveRatio / 10))

