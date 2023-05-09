from evaluation import *
from sklearn.preprocessing import OneHotEncoder
import numpy as np
#________________________________________________________________________________________________________________________________________________________________________
# One Hot Encoding
def oneHotEncoder(data, category):
    # transpose and copy
    dataT = data.T.copy()
    encoder = OneHotEncoder(sparse=False)
    categDict = {}
    # loop through the cats and encode each to OHE
    for i, cat in enumerate(category):
        hotReq = dataT[i]
        hotEncoded = encoder.fit_transform(hotReq.reshape(-1,1))
        for j in encoder.categories_[0]:
            updName = cat+'_'+str(j)
            # update the categDict with the names of the new OHE variables
            if category[cat] == 'categorical':
                categDict[updName] = 'ohe_numerical'
            if category[cat] == 'class':
                categDict[updName] = 'class_numerical'
        dataT = np.append(dataT,hotEncoded.T,axis=0)

    category.update(categDict)
    # copy that excludes categorical and class variables
    categoryListCp = {cat: category[cat] for i, cat in enumerate(category) if category[cat] not in ['categorical', 'class']}
    # updated category dictionary
    dropList = [i for i, cat in enumerate(category) if category[cat] in ['categorical', 'class']]
    dataT = np.delete(dataT,dropList,axis=0)
    return dataT.T, categoryListCp
#________________________________________________________________________________________________________________________________________________________________________

# Normalization 
def normalizeTrain(oheTrainData, category):
    dataTC = oheTrainData.T.copy()
    minMaxVals = []
    for i, attr in enumerate(category):
        if category[attr] == 'numerical':
            #  normalize the attribute using min-max scaling
            colmin, colmax = np.min(dataTC[i]), np.max(dataTC[i])
            minMaxVal = [colmin, colmax]
            dataTC[i] = (dataTC[i] - colmin) / (colmax - colmin)
            minMaxVals.append(minMaxVal)
            # if not numerical
        else:
            minMaxVals.append([0.0, 1.0])
    return dataTC.T, minMaxVals

def normalizealltest(ohe_testdata, category, minMaxVals):
    # normalize a single instance of data
    def normalizeonetest(instance_data, category, minMaxVals):
        for i, attr in enumerate(category):
            # using min-max scaling for numerical attr
            if category[attr] == 'numerical':
                instance_data[i] = (instance_data[i] - minMaxVals[i][0])/(minMaxVals[i][1] - minMaxVals[i][0])
        return instance_data
    result = []
    # normalize all the instances
    for i in ohe_testdata:
        n_ohe_test = normalizeonetest(i, category, minMaxVals)
        result.append(n_ohe_test)
    return np.array(result)
#________________________________________________________________________________________________________________________________________________________________________

def initializeWeights(oheCategory, layerParam, biasTerm=True):
    weightList = []
    inputcategory, outputcategory, inputindex, outputindex = seperateCateg(oheCategory)
    b = int(biasTerm)
    # list of layer parameters that includes bias terms
    biasLayerParam = [len(inputcategory)+b] + list(np.array(layerParam)+b) + [len(outputcategory)]
    # random weight matrices with values between -1 and 1
    weightList = [np.random.rand(biasLayerParam[i+1] - 1 if i != len(biasLayerParam) - 2 else biasLayerParam[i + 1], 
                                        biasLayerParam[i]) * 2 - 1 
                        for i in range(len(biasLayerParam) - 1)]
    return weightList
    
def overallWCost(costList, n, weightLst, lambdaReg):
    def sumOfWeights(weightsList, bias=True):
        if bias:
            # sum of all squared weights, excluding bias
            return np.sum([np.sum(np.square(weight[:, 1:])) for weight in weightsList])
        else:
            # sum of all squared weights, including bias
            return np.sum([np.sum(np.square(weight)) for weight in weightsList])
    # scale it using lambdaRed
    fSum = sumOfWeights(weightLst, bias=0) * lambdaReg / (2 * n)
    costSum = np.sum(costList)
    return costSum / n + fSum

def calulateCost(expectedOutput, actualOutput):
    totalCost = -np.multiply(expectedOutput,np.log(actualOutput)) - np.multiply((1 - expectedOutput), np.log(1 - actualOutput))
    return np.sum(totalCost)
#________________________________________________________________________________________________________________________________________________________________________
# computes the backpropagation deltas for a neural network
def backPropD(predictedOutput, expectedOutput, weightsList, aList, biasTerm=True):
    deltaList = [predictedOutput - expectedOutput]
    # iterate in reverse order
    for i in reversed(range(len(weightsList) - 1)):
        currDeltaLayer = np.dot(weightsList[i+1].T, deltaList[-1]) * aList[i+1] * (1 - aList[i+1])
        if biasTerm:
            # first attr is the bias
            currDeltaLayer[0] = 1
            deltaList.append(currDeltaLayer[1:])
        else:
            # add curr delta to the delta list
            deltaList.append(currDeltaLayer)
    return deltaList[::-1]

#the gradients for a neural network
def gradientD(weightsList, deltaList, attributelist):
    gradList = []
    for i in range(len(weightsList)):
        currAttr = attributelist[i]
        currDelta = np.array([deltaList[i]]).T
        dotProd = currDelta*currAttr
        gradList.append(dotProd)
    return gradList

# computes the deltas for a neural network
def delta(weightLst, aList, expectedVal, actualVal):
    deltaList = [actualVal - expectedVal]
     # iterate in reverse order
    for i in reversed(range(1, len(weightLst))):
        # add curr delta to the delta list
        currDeltaLayer = np.multiply(np.multiply(np.dot(weightLst[i].T, deltaList[-1]), aList[i]), (1-aList[i]))
        deltaList.append(currDeltaLayer[1:])
    return deltaList[::-1]

#performs forward propagation for a neural network
def forwardPropogation(inputdata, weightLst, expectedout):
    # current activation layer
    currALayer = inputdata
    print('CurrALayer at 1 is',currALayer)
    aList = []
    aList.append(currALayer)
    for currLayerIdx, theta in enumerate(weightLst):
        z = np.dot(theta,currALayer)
        a = 1 / (1 + np.exp(-z))
        # add bias term to the current activation layer if needed
        currALayer = np.append(1,a) if (currLayerIdx + 1 != len(weightLst)) else a
        print('CurrALayer at', currLayerIdx + 2, 'is', currALayer)
        aList.append(currALayer)
    result = currALayer
    # requirement print out material
    print('Predicted output for instance', result)
    print('Expected output for instance', expectedout)
    print('Cost, J, associated with instance', calulateCost(expectedout,result))
    return result, calulateCost(expectedout, result), aList
#________________________________________________________________________________________________________________________________________________________________________

# Forward propagation vectorized
# trains a neural network using mini-batch stochastic gradient descent.
def neuralNetwork(normedOheData, oheCategory, weightsList, miniBatchK = 15, lambdaReg = 0.2, lR = 0.01):
    biasTerm = True
    normOheSamp = normedOheData.copy()
    if miniBatchK > len(normedOheData):
        miniBatchK = len(normedOheData)

    np.random.shuffle(normOheSamp)
    # split the shuffled data into mini-batches of size miniBatchK
    splitted = np.array_split(normOheSamp, miniBatchK)
    # separate the input and output categs and their indices
    inputcategory, outputcategory, inputindex, outputindex = seperateCateg(oheCategory)
    
    b = int(biasTerm)
    
    for miniBatch in splitted:
        # transpose
        miniBatch = miniBatch.T
        inputD = miniBatch[inputindex].T
        outputD = miniBatch[outputindex].T

        # forward propagation
        instanceIdx = 0
        totalCost, gradientList = 0, []
        # each instance in the mini-batch
        for one_instance in inputD:
            currALayer = np.append(1,one_instance) if b == 1 else one_instance
            # input layer is the current layer
            outputExpected = outputD[instanceIdx]
            # store the activations including the bias term
            biasAtrr = [currALayer]
            for currLayerIdx, theta in enumerate(weightsList):
                z = np.dot(theta, currALayer)
                # sigmoid activation function
                a = 1 / (1 + np.exp(-z))
                # update acitivations
                currALayer = np.append(1,a) if (b == 1) and (currLayerIdx+1 != len(weightsList)) else a
                biasAtrr.append(currALayer)
            #  final layer is the predicted output
            outputPredicted = currALayer 
            instanceIdx += 1
            # calculate cost
            totalCost += calulateCost(outputExpected,outputPredicted)
            # calculate delta backPropD (back propagation)
            deltaList = backPropD(outputPredicted,outputExpected,weightsList, biasAtrr)
            # calculate the gradients for the layer
            currGradient = gradientD(weightsList,deltaList,biasAtrr)
            gradientList.append(currGradient)

        # gradients from all instances in the mini-batch
        gradientT = [list(x) for x in zip(*gradientList)]
        # regularization to the gradients
        gradientP = [lambdaReg*t for t in weightsList]
        # excluding bias terms 
        for singleP in gradientP:
            singleP[:, 0] = 0
        
        # final val of gradient and minibatch
        gradientDSUM = [np.sum(t,axis=0) for t in gradientT]
        gradientBatch = [(gradientDSUM[i] + gradientP[i])*(1/instanceIdx) for i in range(len(gradientDSUM))]
        allCost = overallWCost(totalCost, instanceIdx+1, weightsList, lambdaReg)
        # update weights
        for i in range(len(weightsList)):
            weightsList[i] -= lR * gradientBatch[i]

    return weightsList, allCost, totalCost 

def trainNeuralNetwork(normedOheTrainingData, oheCategory, layerParam, miniBatchK=15, lambdaReg=0.15, lR=0.01, epsilonB=0.00001, softStop=8000, printQ=False):
    # initialize the weights of the neural network
    initWeights = initializeWeights(oheCategory, layerParam)
    # train the neural network and get the updated weights and cost
    updatedWeights, costSum, pureCost = neuralNetwork(normedOheTrainingData, oheCategory, initWeights, miniBatchK, lambdaReg, lR)
    currentCost = costSum
    smallestCost = costSum
    costList = [currentCost]
    for c in range(softStop):
        if printQ:
            print('Current Cost', currentCost)
            print('Count', c)
        # train the neural network and get the updated weights and cost
        updatedWeights, costSum, pureCost = neuralNetwork(normedOheTrainingData, oheCategory, updatedWeights, miniBatchK, lambdaReg, lR)
        # calculate differnce in cost
        epsilon = costSum - currentCost
        currentCost = costSum
        costList.append(currentCost)
        if currentCost < smallestCost:
            smallestCost = currentCost
        if (epsilon <= epsilonB) and (currentCost < smallestCost):
            break
    return updatedWeights, costList
#________________________________________________________________________________________________________________________________________________________________________
# making predictions using a neural network
def predictNN(testData, oheCateg, weight):
    # predicting the output of a single instance using the given weights
    def predictoneinstance(inputdata, weightLst):
        # add a bias unit to the input data
        currALayer = np.append(1, inputdata)
        aList = [currALayer]
        # claculate activation values for each weight layers
        for i, theta in enumerate(weightLst):
            z = np.dot(theta, currALayer)
            a = 1 / (1 + np.exp(-z))
            currALayer = np.append(1, a) if len(weightLst) - 1 != i else a
            aList.append(currALayer)
        predictedOutput = currALayer
        # if only one output unit convert the predicted output to binary
        if len(predictedOutput) <= 1:
            predictedOutput[0] = 0 if predictedOutput[0] <= 0.5 else 1
        # else convert using OHE
        else:
            predictedOutput[predictedOutput != np.max(predictedOutput)] = 0
            predictedOutput[predictedOutput == np.max(predictedOutput)] = 1
        return predictedOutput, aList[-1]
    # separate the input and output categories and indices from OHE
    inputcategory, outputcategory, inputindex, outputindex = seperateCateg(oheCateg)
    currCount = 0
    predExpecList = []
    # loop through each instance in the test data
    for inst in testData:
        datainput = inst[inputindex]
        expectedOutput = inst[outputindex]
        # predicted output and the activation values of the final layer
        predictedOutput, rawOutput = predictoneinstance(datainput, weight)
        cutExpected = np.where(expectedOutput==1)[0][0]
        cutPredicted = np.where(predictedOutput==1)[0][0]
        predExpecList.append([cutPredicted, cutExpected])
        if cutPredicted == cutExpected:
            currCount += 1
    # accuracy as the fraction of correctly predicted instances
    accuracy = currCount / len(testData)
    return predExpecList, accuracy
#________________________________________________________________________________________________________________________________________________________________________
