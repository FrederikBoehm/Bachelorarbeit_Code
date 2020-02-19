import random
import pandas as pd


# Creates data for the sample learning and validation curve in the thesis

def createSampleData():

    data = _createSampleDataInternal(2000)

    df = pd.DataFrame(data)
    df.to_csv('./sample_data.csv', index=False, sep="\t")

def _createSampleDataInternal(size):
    sample_data = {}
    for i in range(size):
        sample = _createSample()
        sample_sum = sum(sample)

        if sample_sum > (5 + random.random() - 0.5): # Imitates nonlinear data
            _appendToDict(sample_data, sample, 1)
        else:
            _appendToDict(sample_data, sample, 0)

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


def _createSample():
    sample = [] # Need to sum approximatly 100
    for i in range(10):
        rd = random.random()
        sample.append(rd)


    return sample

if __name__ == "__main__":
    createSampleData()