import random
import pandas as pd

def createSampleData():

    data = _createSampleDataInternal(2000)

    df = pd.DataFrame(data)
    df.to_csv('./sample_data.csv', index=False, sep="\t")

def _createSampleDataInternal(size):
    # sample_data = {}
    # for i in range(int(size/2)):
    #     positive_sample = _createPositiveSample()
    #     sample_data = _appendToDict(sample_data, positive_sample, 1)
        
    #     negative_sample = _createNegativeSample()
    #     sample_data = _appendToDict(sample_data, negative_sample, 0)

    # return sample_data
    sample_data = {}
    for i in range(size):
        positive_sample = _createPositiveSample()
        sample_sum = sum(positive_sample)

        if sample_sum > (5 + random.random() - 0.5):
            _appendToDict(sample_data, positive_sample, 1)
        else:
            _appendToDict(sample_data, positive_sample, 0)

    return sample_data

def _appendToDict(dictionary, values, label):
    for index, value in enumerate(values):
        if not f"F{index}" in dictionary:
            dictionary[f"F{index}"] = []

        dictionary[f"F{index}"].append(value)

    if not "LABEL" in dictionary:
        dictionary["LABEL"] = []
    
    dictionary["LABEL"].append(label)
    return dictionary


def _createPositiveSample():
    # positives = [] # Need to sum approximatly 100
    # for i in range(10):
    #     rd = random.random()
    #     positives.append(rd)


    # return positives

    positives = [] # Need to sum approximatly 100
    for i in range(10):
        rd = random.random()
        positives.append(rd)


    return positives

def _createNegativeSample():
    negatives = []
    for i in range(5):
        rd = random.random() * -1
        negatives.append(rd)

    for i in range(5):
        rd = random.random()
        negatives.append(rd)

    random.shuffle(negatives)

    return negatives

if __name__ == "__main__":
    createSampleData()