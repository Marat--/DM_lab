# -*- coding: utf-8 -*-


import math
import numpy as np
import pandas as df

#Class for storying rating
class Rating:
    def __init__(self, userid, movieid, rating):
        self.uid = userid
        self.mid = movieid
        self.rat = rating

def readinratings(fname, arr, splitter="\t"):
        f = open(fname)
        for line in f:
            newline = [int(each) for each in line.split(splitter)]
            userid, movieid, rating = newline[0], newline[1], newline[2]
            arr.append(Rating(userid, movieid, rating))
        arr = sorted(arr, key=lambda rating: (rating.uid, rating.mid))
        return len(arr)
        
def averagerating():
        avg = 0.0
        n = 0
        for i in range(len(trainrats)):
            avg += trainrats[i].rat
            n += 1
        return float(avg/n)      

def dotproduct(v1, v2):
    return sum([v1[i]*v2[i] for i in range(len(v1))])

#Returns the estimated rating corresponding to userid for movieid
#Ensures returns rating is in range [1,5]
def calcrating(uid, mid):
    p = 0.0
    #Sometimes some shit is happening. If you are running a long term tasks
    #you may want to procceed running you code even if an exception has occured
    try:
        p = dotproduct(U[uid], V[mid])
    except:
        print "uid= ", uid
        print "mid= ", mid
        print "p= ", p
        
    if p > 5:
        p = 5
    elif p < 1:         
        p = 1
    return p
        
#Predicts the estimated rating for user with id i for movie with id j
def predict(i, j):
    return calcrating(i, j)


##################### Actual training part ###################################

#Trains the kth column in U and the kth row in V^T. So it trains a certain feature.
def train(k):
    sse = 0.0
    n = 0
    for i in range(len(trainrats)):
        # get current rating
        crating = trainrats[i]
        err = crating.rat - predict(crating.uid, crating.mid)
        sse += err**2
        n += 1
        uTemp = U[crating.uid][k]
        vTemp = V[crating.mid][k]

        U[crating.uid][k] += lrate * (err*vTemp - regularizer*uTemp)
        V[crating.mid][k] += lrate * (err*uTemp - regularizer*vTemp)
    return math.sqrt(sse/n)

def trainratings():        
        # stub -- initial train error
        oldtrainerr = 10000000.0
        
        for k in range(r):    # for each latent feature
            print "k=", k
            
            #oldtrainerr = train(k)
            #An another option is to run first iteration anyway
            #and use is as starting 'trainerr' 
            for epoch in range(maxepochs): 
                trainerr = train(k)
                
                # Check if the train error is still changing.
                # Due to the stochastic nature of steps during the process of finding 
                # minimum of loss function, 'trainerr' is not necessarily always decrease
                # So we need to use ablsolute values here 
                if abs(oldtrainerr-trainerr) < minimprov:
                    break
                oldtrainerr = trainerr
                print "epoch=", epoch, "; trainerr=", trainerr

#Calculates the RMSE using between arr
#and the estimated values in (U * V^T)
def calcrmse(arr):
    sse = 0.0
    total = 0
    for i in range(len(arr)):
        crating = arr[i]
        sse += (crating.rat - calcrating(crating.uid, crating.mid))**2
        total += 1
    return math.sqrt(sse/total)
##############################################################################
 
# These four methods for handling problems with ID in data set  
def countUniqueUsers(arr):
    users = []
    for i in arr:
        users.append(i.uid)
    return len(set(users))
    
def countUniqueMovies(arr):
    movies = []
    for i in arr:
        movies.append(i.mid)
    return len(set(movies))
 
def getMaxMovieIndex(arr):
    maxMid = 0
    for i in arr:
        if i.mid > maxMid:
            maxMid = i.mid
    return maxMid
    
def getMaxUserIndex(arr):
    maxUid = 0
    for i in arr:
        if i.uid > maxUid:
            maxUid = i.uid
    return maxUid

# Class for saving results after varying the parameters
class ResData:
    def __init__(self, parameter, trainrmse, testrmse):
        self.parameter = parameter
        self.trainrmse = trainrmse
        self.testrmse = testrmse        
 
trainrats = []
testrats = []
nusers = 0
nmovies = 0
r = 40 #Rank
lrate=0.001 #Learning rate
regularizer=0.01
minimprov = 0.0001 #Accuracy
maxepochs = 40 #Maximum nuber of epochs per feature
  
if __name__ == "__main__":
    #Make sure that arrays are empty before we fill them up
    trainrats = []
    testrats = []
    readinratings("/home/marat/repo/pysvd/RegSVD/trainrats_1m_clean.csv", trainrats, splitter=",")
    readinratings("/home/marat/repo/pysvd/RegSVD/testrats_1m_clean.csv", testrats, splitter=",")

    #Make a first guess
    avg = averagerating()    
    initval = math.sqrt(avg/r)
    #initval = 0.0001
    
    #This magic is nesseary only because data sets are not clean in terms of ID
    nusers = max([countUniqueUsers(trainrats)+1, getMaxUserIndex(trainrats)+1])
    nmovies = max([countUniqueMovies(trainrats)+1, getMaxMovieIndex(trainrats)+1])    
        
    """ Here we are varying regularization paremeter
    # Be careful, "regularizer" is a global variable
    resultData = []
    regularizer_start = 0.0
    regularizer_end = 0.1
    regularizer = regularizer_start
    while regularizer <= regularizer_end:
        print "reg: ", regularizer
        U = [[initval]*r for i in range(nusers)]
        V = [[initval]*r for i in range(nmovies)]
        trainratings()
        trainRMSE = calcrmse(trainrats)
        testRMSE = calcrmse(testrats)
        resultData.append(ResData(regularizer, trainRMSE, testRMSE))
        regularizer += 0.001
    """
    
    """ Here we are varing rank of the matrixes
    # Be careful, "r" is a global variable 
    resultData = []
    r_start = 5
    r_end = 70
    r = r_start
    while r <= r_end:
        print "r: ", r
        initval = math.sqrt(avg/r)
        U = [[initval]*r for i in range(nusers)]
        V = [[initval]*r for i in range(nmovies)]
        trainratings()
        trainRMSE = calcrmse(trainrats)
        testRMSE = calcrmse(testrats)
        resultData.append(ResData(r, trainRMSE, testRMSE))
        r += 1
    """
    
    """ Here we are saving the result of varying the parameters    
    import csv
    with open('/home/marat/repo/pysvd/RegSVD/result4.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerows((d.parameter, d.trainrmse, d.testrmse) for d in resultData)
    """
    
    # Initialize user and movie matrixes with the first guess
    U = [[initval]*r for i in range(nusers)]
    V = [[initval]*r for i in range(nmovies)]
    trainratings()

    # calculate final performance
    print "rmsetrain: ", calcrmse(trainrats)
    print "rmseTEST: ", calcrmse(testrats)
