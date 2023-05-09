import numpy as np
import matplotlib.pyplot as plt
import csv
#________________________________________________________________________________________________________________________________________________________________________

# Extraction and Seperation 

def extractData(name:str, delimit:str, classN:str, categ:str):
    file = open("datasets/" + name, encoding = 'utf-8-sig')
    reader = csv.reader(file, delimiter = delimit)
    data = []
    for row in reader:
        data.append(row)
    fileCateg = {}
    for i in data[0]:
        fileCateg[i] = categ
    fileCateg[classN] = 'class'
    fileData = np.array(data[1:]).astype(float)
    return fileData, fileCateg

def seperateCateg(ohecateg):
    inputCateg, outputCateg = [],[]
    inputIdx, outputIdx = [],[]
    for i, categ in enumerate(ohecateg):
        if ohecateg[categ] != 'class_numerical':
            inputCateg.append(categ) 
            inputIdx.append(i) 
        else: #class
            outputCateg.append(categ)  
            outputIdx.append(i) 
    return inputCateg, outputCateg, inputIdx, outputIdx
#________________________________________________________________________________________________________________________________________________________________________

# Evaluation Metrics

def accuracy(truePosi, trueNega, falsePosi, falseNega):
	return (truePosi + trueNega) / (truePosi + trueNega + falseNega + falsePosi)

def precision(truePosi, falsePosi):
	if (truePosi + falsePosi) == 0:
		return 0
	return truePosi / (truePosi + falsePosi)

def recall(truePosi, falseNega):
	if (truePosi + falseNega) == 0:
		return 0
	return truePosi / (truePosi + falseNega)

def fscore(truePosi, falsePosi, falseNega, beta = 1):
    if not (truePosi or falsePosi or falseNega):
        return 0
    
    precision = truePosi / (truePosi + falsePosi) if truePosi + falsePosi > 0 else 0
    recall = truePosi / (truePosi + falseNega) if truePosi + falseNega > 0 else 0
    
    if precision == 0 and recall == 0:
        return 0
    
    fScore = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall) if precision + recall > 0 else 0
    return fScore
#________________________________________________________________________________________________________________________________________________________________________

def evalResults(results, corrResult, beta=1):
    accList, preList, recList, f1List = [], [], [], []
    for res in results:
        truePosi = sum(p == a == corrResult for p, a in res)
        trueNega = sum(p == a != corrResult for p, a in res)
        falsePosi = sum(p == corrResult and a != corrResult for p, a in res)
        falseNega = sum(p != corrResult and a == corrResult for p, a in res)
        accList.append(accuracy(truePosi, trueNega, falsePosi, falseNega))
        preList.append(precision(truePosi, falsePosi))
        recList.append(recall(truePosi, falseNega))
        f1List.append(fscore(truePosi, falsePosi, falseNega, beta))
    return accList, preList, recList, f1List

def meanEval(results, corrResult, beta=1):
    accuarcylists, precisionlists, recalllists, fscorelists = evalResults(results, corrResult, beta)
    return sum(accuarcylists)/len(accuarcylists), sum(precisionlists)/len(precisionlists), sum(recalllists)/len(recalllists), sum(fscorelists)/len(fscorelists)
#________________________________________________________________________________________________________________________________________________________________________