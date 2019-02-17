import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
from loadData import LoadData




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', required=True)
    io_args = parser.parse_args()
    modelName = io_args.model

    if modelName == "CNN":
        # TODO: CODE FOR CALLING CNN CLASS GOES IN HERE

        dataFrame = LoadData()
        data = dataFrame.loadData()

        print(data)

        

    elif modelName == "lin-reg":
        # TODO: CODE FOR CALLING LINEAR REGRESSION CLASS GOES IN HERE
        print("CODE FOR")

   
        
    else:
        print("Unknown model: %s" % model)