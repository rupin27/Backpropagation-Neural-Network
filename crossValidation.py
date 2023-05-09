from neuralNetwork import *
from evaluation import *
#________________________________________________________________________________________________________________________________________________________________________
# stratified k-fold cross validation
def stratifiedKFold(data, categDict, k=10):
    classIndex = list(categDict.values()).index("class")
    # getting all the unique classes
    classes = np.unique(data[:,classIndex])
    nClass = len(classes)
    # splitting the data based on class
    listClasses = [data[data[:, classIndex] == c] for c in classes]
    # into k folds
    listClasses = [np.random.permutation(c) for c in listClasses]
    splittedList = [np.array_split(c, k) for c in listClasses]
    # combing for the whole dataset
    combinedList = [np.concatenate([splittedList[i][j] for i in range(nClass)]) for j in range(k)]
    return combinedList

def kfoldcrossvalidneuralnetwork(data, category, layerParam, k = 10, miniBatchK = 15, lambdaReg = 0.15, lR = 0.01, epsilonB = 0.00001, softStop = 6000, printQ = False):
    def oheStratifiedKFold(oheData, categDict, k=10):
         # index of the column representing the class variable
        classIndex = [i for i, j in enumerate(categDict.values()) if j == "class_numerical"][0]
        # unique classes in the data and split the data
        classes = np.unique(oheData[:, classIndex])
        classesList = [oheData[oheData[:, classIndex] == c] for c in classes]
        classesList = [np.random.permutation(c) for c in   classesList]
        # shuffle data & split it into k folds
        splits = [np.array_split(c, k) for c in classesList]
        combList = [np.concatenate([splits[i][j] for i in range(len(classes))]) for j in range(k)]
        return combList
    # perform one-hot encoding on the input data and categs
    oheData, oheCateg = oneHotEncoder(data, category)
    folded = oheStratifiedKFold(oheData, oheCateg, k)
    pelist, accList, costList = [], [], []
    # looping over folds
    for i in range(k):
        print('Fold',i+1,'Training in Progress')
        if printQ:
            print('Fold',i+1)
        oheTestFolds, oheTrainFolds = folded[i].copy(), np.vstack(folded[:i] + folded[i + 1:])
        # normalize
        normOheTrain, minMax = normalizeTrain(oheTrainFolds, oheCateg)
        normOheTest = normalizealltest(oheTestFolds, oheCateg, minMax)
        finalWeight, costlist = trainNeuralNetwork(normOheTrain, oheCateg, layerParam, miniBatchK, lambdaReg, lR, epsilonB, softStop, printQ)
        # get predictions based on training
        pEVal, accVal = predictNN(normOheTest, oheCateg, finalWeight)
        print('Fold',i+1,'Training Completed, Accuracy = ', accVal)
        pelist.append(pEVal)
        accList.append(accVal)
        costList.append(costlist)
    averageAcc = np.mean(accList)
    return pelist, averageAcc, costList
#________________________________________________________________________________________________________________________________________________________________________