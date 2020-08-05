# Perceptron implementation
import util
import random


class PerceptronClassifier:
    """
  Perceptron classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details.

    Use the provided self.weights[label] datastructure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

        self.features = trainingData[0].keys()  # could be useful later

        for i in range(len(self.weights)):
            for j in range(len(self.features)):
                # print self.features[i]
                self.weights[i][self.features[j]] = random.randint(0, 1)

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                label_max = []
                for j in self.legalLabels:
                    import util
                    label_max.append(self.weights[j] * trainingData[i])

                label_final = max(label_max)
                indexvalue = label_max.index(label_final)

                if (label_final != trainingLabels[i]):
                    self.weights[indexvalue] = self.weights[indexvalue] - trainingData[i]
                    self.weights[trainingLabels[i]] = self.weights[trainingLabels[i]] + trainingData[i]

    def classify(self, data):
        """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.

    Recall that a datum is a util.counter... Do not modify this method.
    """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighOddsFeatures(self, class1, class2):
        """
    Returns:
    featuresClass1 -- the 100 largest weight features for class1 (as a list)
    featuresClass2 -- the 100 largest weight features for class2
    featuresOdds -- the 100 best features for difference in feature values
                     w_class1 - w_class2

    """

        featuresClass1 = []
        featuresClass2 = []
        featuresOdds = []

        ## Your code here

        return featuresClass1, featuresClass2, featuresOdds

    def findHighWeightFeatures(self, label):
        """
    Returns a list of the 100 features with the greatest weight for some label
    """
        featuresWeights = []

        for keys in range(100):
            featuresWeights.append(self.weights[label].sortedKeys()[keys])

        return featuresWeights
