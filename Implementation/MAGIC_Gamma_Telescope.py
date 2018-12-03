import pandas as pd
import numpy as np
from numpy import *
import time
from scipy import stats



def isLeaf(node):
    return False if isinstance(node,Node) else True

def isNode(node):
    return isinstance(node, Node)

def calAccuracy(pred, label):

    acc = 0.0
    for i in range(len(pred)):
        if (pred[i] == label[i][-1]):
            acc += 1
    return acc / len(pred)

def classify_sample(node, test_sample):
    if isLeaf(node):
        return np.argmax(node)
    C = 0
    if (test_sample[node.split_index] < node.split_value):
        C = classify_sample(node.left, test_sample)
    else:
        C = classify_sample(node.right, test_sample)

    return C

def classify_data(node, data):
    result = np.zeros(len(data))
    for i in range(len(data)):
        sample = data[i]
        result[i] = classify_sample(node, sample)

    return result

class Node:
    def __init__(self):
        self.split_index = None
        self.split_value = None
        self.left = None
        self.right = None

    def getMean(self):
        if isNode(self.left):
            self.left = self.left.getMean()

        if isNode(self.right):
            self.right = self.right.getMean()

        return (self.left + self.right) / (2.0)

def Gini(data):

    numOfG = data[:,-1].sum()
    pG = numOfG / len(data)
    pH = 1 - pG

    giniG = 1 - pG * pG
    giniH = 1 - pH * pH


    return giniG + giniH

def ShannoEnt(data):
    numOfG = data[:, -1].sum()
    pG = numOfG / len(data)
    pH = 1 - pG

    if (pH == 0 or pG == 0):
        return 0
    shannonEnt = -pH * np.log2(pH) - pG * np.log2(pG)

    return shannonEnt

def BinarySplit(data, featIndex, splitVal):
    subData0 = data[data[:, featIndex] < splitVal]
    subData1 = data[data[:, featIndex] >= splitVal]

    return subData0, subData1

###################################################################################################

def chooseBestSplit_GR(data):

    baseS = ShannoEnt(data)

    # print("S = ", S)

    n = data.shape[1]; bestGR = -inf; bestIndex = 0; bestSplitVal = 0

    for featIndex in range(n - 1):
        for splitVal in np.percentile(data[:, featIndex], [10, 20, 30, 40, 50, 60, 70, 80, 90]):

            subData0, subData1 = BinarySplit(data, featIndex, splitVal)

            if (len(subData0) == 0 or len(subData1) == 0):
                continue

            total = len(subData0) + len(subData1)
            p0 = len(subData0) / len(data)
            p1 = len(subData1) / len(data)

            newS = ShannoEnt(subData0) * p0 + ShannoEnt(subData1) * p1

            splitInfo = - p0 * np.log2(p0) - p1 * np.log2(p1)

            GR = (baseS - newS) / splitInfo

            if (GR > bestGR):
                bestGR = GR
                bestIndex = featIndex
                bestSplitVal = splitVal

    # check valid split again and return bestIndex, bestSplitValue

    subData0, subData1 = BinarySplit(data, bestIndex, bestSplitVal)

    if (len(subData0) == 0 or len(subData1) == 0):
        return None, 0

    return bestIndex, bestSplitVal

###################################################################################################

def NewS(subData0, subData1, criteria):

    if (len(subData0) == 0 or len(subData1) == 0):
        return inf

    total = len(subData0) + len(subData1)
    p0 = len(subData0) / total
    p1 = len(subData1) / total

    if criteria == 'VA':
        newS = np.var(subData0[:, -1]) * p0 + np.var(subData1[:, -1]) * p1

    elif criteria == 'GI':
        newS = Gini(subData0) * p0 + Gini(subData1) * p1
    else:
        newS = ShannoEnt(subData0) * p0 + ShannoEnt(subData1) * p1

    return newS

def BaseS(data, criteria):

    if criteria == 'VA':
        S = np.var(data[:,-1])
    elif criteria == 'GI':
        S = Gini(data)
    else:
        S = ShannoEnt(data)

    return S


def chooseBestSplit_VA_GI_IG(data, criteria):

    S = BaseS(data, criteria)

    # print("S = ", S)

    n = data.shape[1]; bestS = inf; bestIndex = 0; bestSplitVal = 0

    for featIndex in range(n-1):
        for splitVal in np.percentile(data[:,featIndex], [10, 20, 30, 40, 50, 60, 70, 80, 90]):

            subData0, subData1 = BinarySplit(data, featIndex, splitVal)

            newS = NewS(subData0, subData1, criteria)

            if (newS < bestS):
                bestS = newS
                bestIndex = featIndex
                bestSplitVal = splitVal

    # print("bestS = ", bestS)

    if (S - bestS) < 0:
        return None, 0

    # check valid split again and return bestIndex, bestSplitValue

    subData0, subData1 = BinarySplit(data, bestIndex, bestSplitVal)

    if (len(subData0) == 0 or len(subData1) == 0):
        return None, 0

    return bestIndex, bestSplitVal

###################################################################################################

###################################################################################################


class DecisionTree:
    def __init__(self, train_data, criteria):
        self.train_data = train_data
        self.criteria = criteria
        self.root = self.induction(train_data, criteria)

    ###################################################################################################


    ###################################################################################################

    def induction(self, data, criteria):

        numOfG = data[:, -1].sum()
        freqClasses = np.array([len(data) - numOfG, numOfG])

        if (data[:, -1].sum() == data.shape[0]):
            return freqClasses

        if criteria == 'GR':
            bestIndex, bestSplitVal = chooseBestSplit_GR(data)
        else:
            bestIndex, bestSplitVal = chooseBestSplit_VA_GI_IG(data, criteria)

        if (bestIndex is None):
            return freqClasses

        newNode = Node()
        newNode.split_index = bestIndex
        newNode.split_value = bestSplitVal

        subDataLeft, subDataRight = BinarySplit(data, bestIndex, bestSplitVal)

        newNode.left = self.induction(subDataLeft, criteria)
        newNode.right = self.induction(subDataRight, criteria)

        return newNode

    ###################################################################################################

    ###################################################################################################

    def postPrune(self, node, prune_data):

        if (isLeaf(node)):
            return node

        if (len(prune_data) == 0):
            return node.getMean()

        subLeft, subRight = BinarySplit(prune_data, node.split_index, node.split_value)

        if (isNode(node.left)):
            node.left = self.postPrune(node.left, subLeft)

        if (isNode(node.right)):
            node.right = self.postPrune(node.right, subRight)

        if (isLeaf(node.left) and isLeaf(node.right)):

            predictNoMerge = classify_data(node, prune_data)
            noMergeAcc = calAccuracy(predictNoMerge, prune_data)

            mergeNode = node.getMean()
            predictMerge = classify_data(mergeNode, prune_data)
            mergeAcc = calAccuracy(predictMerge, prune_data)

            if (mergeAcc > noMergeAcc):
                return mergeNode

        return node

    def prune(self, prune_data):

        self.root = self.postPrune(self.root, prune_data)

    def predict(self, test_data):
        return classify_data(self.root, test_data)

    def test(self, test_data):
        predict = self.predict(test_data)
        return calAccuracy(predict, test_data)

    def error(self, test_data):
        return 1 - self.test(test_data)

###################################################################################################

class Ensembled_DT:

    def __init__(self, train_data):
        self.M_IG = DecisionTree(train_data, 'IG')
        self.M_GR = DecisionTree(train_data, 'GR')
        self.M_VA = DecisionTree(train_data, 'VA')

    def predict(self, test_data):

        predict_IG = self.M_IG.predict(test_data)
        predict_GR = self.M_GR.predict(test_data)
        predict_VA = self.M_VA.predict(test_data)

        result = predict_IG + predict_GR + predict_VA
        result[result < 2] = 0
        result[result >= 2] = 1

        return result

    def prune(self, prune_data):
        self.M_IG.prune(prune_data);
        self.M_GR.prune(prune_data);
        self.M_VA.prune(prune_data);

    def test(self, test_data):
        predict = self.predict(test_data)
        return calAccuracy(predict, test_data)

    def error(self, test_data):
        return 1 - self.test(test_data)

###################################################################################################

def ten_fold_cross_validation():
    n_folds = 10

    indices = np.random.permutation(data.shape[0])

    fold_len = int(len(data) / n_folds)
    maskFold = np.ones(data.shape[0], dtype=bool)

    startIdx = 0
    endIdx = fold_len

    x = np.zeros(n_folds)
    y = np.zeros(n_folds)

    for i in range(n_folds):

        if (i == n_folds - 1):
            endIdx = len(data)

        maskFold[startIdx:endIdx] = False
        train_data = data[maskFold]
        test_data = data[~maskFold]

        M_star = Ensembled_DT(train_data)

        err_M_star = M_star.error(test_data)

        M_GI = DecisionTree(train_data, 'GI')

        err_M_GI = M_GI.error(test_data)

        x[i] = err_M_star
        y[i] = err_M_GI

        # print(err_M_star, " ", err_M_GI)


        maskFold[startIdx:endIdx] = True
        startIdx = endIdx
        endIdx += fold_len

    x1 = x.mean()
    x2 = y.mean()
    var1 = x.var() # s1 = x.std()
    var2 = y.var() # s2 = y.std()

    t_val = np.abs(x1 - x2) / np.sqrt( (var1 + var2) / n_folds )

    # sig = 0.05
    # df = 9
    # from table we have

    critical_value = 2.26

    if (t_val < critical_value):
        print("There is no statistically significant difference between two classifiers")
    else:
        print("There is a statistically significant difference between two classifiers")

###################################################################################################


df = pd.read_csv("magic04.data", header=None)

# preprocess

data = df.values

maskH = (data[:,-1] == 'h')
data[:,-1][maskH] = 0

maskG = (data[:,-1] == 'g')
data[:,-1][maskG] = 1

###################################################################################################

ten_fold_cross_validation()