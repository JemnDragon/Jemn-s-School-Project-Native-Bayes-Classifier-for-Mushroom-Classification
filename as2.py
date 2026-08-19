#    NAME: Jeremiah Monteverde, 5007922369, CS 422 - 1001,
#          Assignment 2
#    DESCRIPTION: An assignment focused on NBC model!
#    INPUT: The mushroom dataset from kaggle!
#    OUTPUT: Evaluation metrics for both test and training sets

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Naive Bayes Classifier class for categorical inputs.
class NB:

    # model fitting function!
    # assuming that:
        #   X is np.array of training data
        #   y is np.array of training data's classes
    def fit(self, X_train, y_train):
        # grab # of examples(exa) & # of inputs(inp)
        self.exa, self.inp = X_train.shape
        self.lh = np.zeros((self.exa, self.inp))
        
        # grab all unique classes from y_train
        self.uniqueY = np.unique(y_train)
        self.exa = len(self.uniqueY)

        # finds count of every class appearance
        yCounts = []
        for c in self.uniqueY:  # for every class label of data set
            count = np.sum(y_train == c)    # adds sum of all times c appears in y_train
            yCounts.append(count)           # adds count to list
        self.yCounts = np.array(yCounts)

        # calculates prior probabilities
        self.prProb = self.yCounts / self.exa

        # iterating through all examples
        for i, c in enumerate(self.uniqueY):
            xExas = X_train[y_train == c]

            # calculates likelihoods
            exaSum = np.sum(xExas, axis=0)
            exaCount = xExas.shape[0]
            self.lh[i] = exaSum / exaCount
    
    # function to predict class of a given example
    #   assuming x is a 1d array example 
    def predict_class(self, x):
        x = np.array(x, dtype=float)

        # empty list for posteriors later
        post = []

        for i, c in enumerate(self.uniqueY):
            pr = np.log(self.prProb[i])

            # calculates log of true/false probabilities
            epsVar = 1e-10
            trueProb = np.log(self.lh[i] + epsVar)
            falseProb = np.log(1 - self.lh[i] + epsVar)

            # finds log prob for specific input given class c
            xLh = x * trueProb + (1-x) * falseProb
            postVal = pr + np.sum(xLh)    # calculates posterior

            post.append(postVal)    # adds to posterior list

        # returns class label with highest posterior prob
        return self.uniqueY[np.argmax(post)]
    
    # function to find probability estimates for the positive class
    #   for each example
    # assuming dSet is a 2d array representing data set
    def class_probs(self, dSet):
        # empty list for probabilities
        probs = []

        # iterating through every example in x
        for xExa in dSet:
            # similar process to one in classProb
            poster = []
            for i, c in enumerate(self.uniqueY):
                pr = np.log(self.prProb[i])

                epsVar = 1e-10
                trueProb = np.log(self.lh[i] + epsVar)       # log(P(x = 1|y=c))
                falseProb = np.log(1 - self.lh[i] + epsVar)  # log(P(x=0|y=c))

                # finds log prob for specific input given class c
                xLh = xExa * trueProb + (1-xExa) * falseProb
                postVal = pr + np.sum(xLh)    # calculates posterior

                poster.append(postVal)    # adds to posterior list

            # convert log posterior to probabilities
            mPoster = np.max(poster)
            ePoster = np.exp(np.array(poster) - mPoster)    
                # removing max to avoid potential overflow issues
            normProbs = ePoster / np.sum(ePoster)  # normalizing probability!
            probs.append(normProbs)

        return np.array(probs)

    # predict class function for given data!
    # assuming that:
    #   X_test is 2d array representing data set
    def predict(self, X_test):
        predY = []

        for xExa in X_test:
            prediction = self.predict_class(xExa)
            predY.append(prediction)
        
        return np.array(predY)  # returns predicted classes!

# evaluation metrics function
#   given the:
#       data set, data set's target classes, positive class,
#       predicted classes for dataset,class probabilities for 
#       dataset, and unique classes to model!!
def evaMetrics(X, y, pos, predY, probsY, uniqueY):
    # calculates the accuracy
    acc = np.mean(y == predY)
    print("  Accuracy = ", acc)

    # calcualtes the true/false positives and true/false negatives
    tp = np.sum((y == pos) & (predY == pos))
    fn = np.sum((y == pos) & (predY != pos))
    tn = np.sum((y != pos) & (predY != pos))
    fp = np.sum((y != pos) & (predY == pos))

    # calculates sensitivity
    sens = 0
    if (tp + fn) > 0:
        sens = tp / (tp + fn)
    print("  Sensitivity = ", sens)

    # calculates specificity
    speci = 0
    if (tn+fp) > 0:
        speci = tn / (tn + fp)
    print("  Specificity = ", speci)

    # calculates precision
    prec = 0
    if (tp + fp) > 0:
        prec = tp / (tp + fp)
    print("  Precision = ", prec)

    # calculates f1-score
    f1_sco = 0
    if (sens + prec) > 0:
        f1_top = sens * prec
        f1_bottom = sens + prec
        f1_sco = 2 * (f1_top / f1_bottom)
    print("  F1-Score = ", f1_sco)

    # calculates the log loss
    yArray = (y == pos).astype(int)

    pIndex = np.where(uniqueY == pos)[0][0]
    predProb = probsY[:, pIndex]
    epsVar = 1e-15
    predProb = np.clip(predProb, epsVar, 1 - epsVar)

    ll = -np.mean(yArray * np.log(predProb) + (1 - yArray) * np.log(1-predProb))
    print("  LogLoss = ", ll)

def main() :
    # grabbing the dataset to store into a panda
    baseData = pd.read_csv("mushrooms.csv")

    # note: the binary outputs is either 'p' for poisonous
    #           and 'e' for edible

    # using panda library function to encode them!
    encodedData = pd.get_dummies(baseData, drop_first=1, dtype=int)

    #print("Panda Contents:\n ", baseData)
    #print("\nPanda Contents:\n ", encodedData)

    # setting training data (X), and its labels (y)
    X = encodedData.iloc[:, 1:].astype(int)
    y = encodedData.iloc[:, 0].astype(int)

    # split the data into training and testing sets!
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.2, random_state=42) 

    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype = float)
    
    model = NB()
    model.fit(X_train, y_train)

    uniqueClasses = np.unique(y)  
    posClass = 1

    # grab predictions + probs for training set
    pTrain = model.predict(X_train)
    probTrain = model.class_probs(X_train)

    # grab predictions + probs for test set
    pTest = model.predict(X_test)
    probTest = model.class_probs(X_test)

    # doing evaluation metrics for training set!
    print("Training Set's Evaluation Metrics")
    evaMetrics(X_train, y_train, posClass, pTrain, probTrain, uniqueClasses)

    # doing evaluation metrics for test set!
    print("\n Test Set's Evaluation Metrics")
    evaMetrics(X_test, y_test, posClass, pTest, probTest, uniqueClasses)


if __name__ == "__main__" :
    main()