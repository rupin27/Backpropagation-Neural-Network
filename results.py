
from evaluation import *
from neuralNetwork import *
from crossValidation import *
import matplotlib.pyplot as plt

housedata, housecategory = extractData('hw3_house_votes_84.csv', ',', "class", 'categorical')
winedata, winecategory = extractData('hw3_wine.csv', '\t', "# class", 'numerical')
cancerdata, cancercategory = extractData('hw3_cancer.csv', '\t', "Class", "numerical")

"""
_______________________________________________________________________________________________________________________________________
### House Data Analysis
Trained with MiniBatch, vectorized neural network, divided to 2 group, (batchsize = 435/2 = 217). Done by the virtue of the miniBatchK variable.
"""

hiddenLayerParam = [4,4,4]
epochSize = 1000
outputList_1a, acc_1a, costList_1a = kfoldcrossvalidneuralnetwork(housedata, housecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuarcy_1a, precision_1a, recall_1a, fscore_1a= meanEval(outputList_1a, 1)
print("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", acc_1a)
print("fScore:", fscore_1a)
plt.plot(range(epochSize+1), costList_1a[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [4,4]
epochSize = 1000
outputList_1b, acc_1b, costList_1b = kfoldcrossvalidneuralnetwork(housedata, housecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuarcy_1b, precision_1b, recall_1b, fscore_1b= meanEval(outputList_1b, 1)
print("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", acc_1b)
print("fScore:", fscore_1b)
plt.plot(range(epochSize+1), costList_1b[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [2]
epochSize = 1000
outputList_1c, acc_1c, costList_1c = kfoldcrossvalidneuralnetwork(housedata, housecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuarcy_1c, precision_1c, recall_1c, fscore_1c= meanEval(outputList_1c, 1)
print("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", acc_1c)
print("fScore:", fscore_1c)
plt.plot(range(epochSize+1), costList_1c[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [4]
epochSize = 1000
outputList_1d, acc_1d, costList_1d = kfoldcrossvalidneuralnetwork(housedata, housecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuarcy_1d, precision_1d, recall_1d, fscore_1d= meanEval(outputList_1d, 1)
print("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", acc_1d)
print("fScore:", fscore_1d)
plt.plot(range(epochSize+1), costList_1d[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [8,8]
epochSize = 1000
outputList_1e, acc_1e, costList_1e = kfoldcrossvalidneuralnetwork(housedata, housecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuarcy_1e, precision_1e, recall_1e, fscore_1e= meanEval(outputList_1e, 1)
print("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", acc_1e)
print("fScore:", fscore_1e)
plt.plot(range(epochSize+1), costList_1e[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [8]
epochSize = 1000
outputList_1f, acc_1f, costList_1f = kfoldcrossvalidneuralnetwork(housedata, housecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuarcy_1f, precision_1f, recall_1f, fscore_1f= meanEval(outputList_1f, 1)
print("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", acc_1f)
print("fScore:", fscore_1f)
plt.plot(range(epochSize+1), costList_1f[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [16,16]
epochSize = 1000
outputList_1g, acc_1g, costList_1g = kfoldcrossvalidneuralnetwork(housedata, housecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuarcy_1g, precision_1g, recall_1g, fscore_1g= meanEval(outputList_1g, 1)
print("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", acc_1g)
print("fScore:", fscore_1g)
plt.plot(range(epochSize+1), costList_1g[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [16,16, 16]
epochSize = 1000
outputList_1h, acc_1h, costList_1h = kfoldcrossvalidneuralnetwork(housedata, housecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuarcy_1h, precision_1h, recall_1h, fscore_1h= meanEval(outputList_1h, 1)
print("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", acc_1h)
print("fScore:", fscore_1h)
plt.plot(range(epochSize+1), costList_1h[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("House Votes Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

"""
_______________________________________________________________________________________________________________________________________
### Wine Data Analysis
Trained with MiniBatch, vectorized neural network, divided to 2 group, (batchsize = 435/2 = 217). Done by the virtue of the miniBatchK variable.
"""

hiddenLayerParam = [4,4,4]
epochSize = 1000
outputList_2a, acc_2a, costList_2a = kfoldcrossvalidneuralnetwork(winedata, winecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuracy_2a, precision_2a, recall_2a, fscore_2a= meanEval(outputList_2a, 1)
print("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", float("{0:.4f}". format(acc_2a)))
print("fScore:", float("{0:.4f}". format(fscore_2a)))
plt.plot(range(epochSize+1), costList_2a[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [4,4]
epochSize = 1000
outputList_2b, acc_2b, costList_2b = kfoldcrossvalidneuralnetwork(winedata, winecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuracy_2b, precision_2b, recall_2b, fscore_2b= meanEval(outputList_2b, 1)
print("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", float("{0:.4f}". format(acc_2b)))
print("fScore:", float("{0:.4f}". format(fscore_2b)))
plt.plot(range(epochSize+1), costList_2b[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [2]
epochSize = 1000
outputList_2c, acc_2c, costList_2c = kfoldcrossvalidneuralnetwork(winedata, winecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuracy_2c, precision_2c, recall_2c, fscore_2c= meanEval(outputList_2c, 1)
print("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", float("{0:.4f}". format(acc_2c)))
print("fScore:", float("{0:.4f}". format(fscore_2c)))
plt.plot(range(epochSize+1), costList_2c[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [4]
epochSize = 1000
outputList_2d, acc_2d, costList_2d = kfoldcrossvalidneuralnetwork(winedata, winecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuracy_2d, precision_2d, recall_2d, fscore_2d= meanEval(outputList_2d, 1)
print("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", float("{0:.4f}". format(acc_2d)))
print("fScore:", float("{0:.4f}". format(fscore_2d)))
plt.plot(range(epochSize+1), costList_2d[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [8, 8]
epochSize = 1000
outputList_2e, acc_2e, costList_2e = kfoldcrossvalidneuralnetwork(winedata, winecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuracy_2e, precision_2e, recall_2e, fscore_2e= meanEval(outputList_2e, 1)
print("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", float("{0:.4f}". format(acc_2e)))
print("fScore:", float("{0:.4f}". format(fscore_2e)))
plt.plot(range(epochSize+1), costList_2e[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [8]
epochSize = 1000
outputList_2f, acc_2f, costList_2f = kfoldcrossvalidneuralnetwork(winedata, winecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuracy_2f, precision_2f, recall_2f, fscore_2f= meanEval(outputList_2f, 1)
print("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", float("{0:.4f}". format(acc_2f)))
print("fScore:", float("{0:.4f}". format(fscore_2f)))
plt.plot(range(epochSize+1), costList_2f[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [16, 16]
epochSize = 1000
outputList_2g, acc_2g, costList_2g = kfoldcrossvalidneuralnetwork(winedata, winecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuracy_2g, precision_2g, recall_2g, fscore_2g= meanEval(outputList_2g, 1)
print("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", float("{0:.4f}". format(acc_2g)))
print("fScore:", float("{0:.4f}". format(fscore_2g)))
plt.plot(range(epochSize+1), costList_2g[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()

hiddenLayerParam = [16, 16, 16]
epochSize = 1000
outputList_2h, acc_2h, costList_2h = kfoldcrossvalidneuralnetwork(winedata, winecategory, hiddenLayerParam, k = 10, miniBatchK = 2, lambdaReg = 0.1, lR = 0.1, epsilonB = 0.0001, softStop = epochSize, printQ = False)
accuracy_2h, precision_2h, recall_2h, fscore_2h= meanEval(outputList_2h, 1)
print("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
print("Accuracy:", float("{0:.4f}". format(acc_2h)))
print("fScore:", float("{0:.4f}". format(fscore_2h)))
plt.plot(range(epochSize+1), costList_2h[1])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Wine Dataset Neural Network Training with " + str(hiddenLayerParam) + " hidden layers and " + str(epochSize) + " epochs")
plt.show()