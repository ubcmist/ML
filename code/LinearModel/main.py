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
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0
        
        # YOUR CODE HERE FOR Q1.1.1
        most_stars = item_inverse_mapper[np.argmax(np.sum(X,axis=0))]
        print("Item with most stars:", url_amazon % most_stars)
        print("Number of stars:", np.max(np.sum(X,axis=0)))

        # YOUR CODE HERE FOR Q1.1.2
        most_reviewed = user_inverse_mapper[np.argmax(np.sum(X_binary,axis=1))]
        print("User with most reviews:", most_reviewed)
        print("Number of reviews for that user:", np.max(np.sum(X_binary,axis=1)))

        # YOUR CODE HERE FOR Q1.1.3
        plt.figure()
        plt.hist(X.getnnz(axis=1));
        plt.yscale('log', nonposy='clip')
        plt.title('Ratings per user');
        plt.savefig(os.path.join("..", "figs", "q1_ratings_per_user.pdf"))

        plt.figure()
        plt.hist(X.getnnz(axis=0));
        plt.yscale('log', nonposy='clip');
        plt.title('Ratings per item');
        plt.savefig(os.path.join("..", "figs", "q1_ratings_per_item.pdf"))

        plt.figure()
        plt.hist(np.array(X[X!=0]).flatten(), bins=np.arange(0.5,6));
        plt.title("All ratings");
        plt.savefig(os.path.join("..", "figs", "q1_total_ratings.pdf"))

    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]

        print(url_amazon % grill_brush)

        # YOUR CODE HERE 
        def find_nn(model, X, query_ind):
            model.fit(X) 
            X_query = X[query_ind] if X[query_ind].ndim==2 else X[query_ind][None] # nonsense needed for non-sparse X
            _, inds = model.kneighbors(X_query) 
            return [ind for ind in inds[0] if ind != query_ind] # don't return yourself as a neighbour

        euc_items = find_nn(NearestNeighbors(n_neighbors=6), X.T, grill_brush_ind)

        print("Similar items using Euclidean distance:")
        for i in euc_items:
            print("  ", url_amazon % item_inverse_mapper[i])

        X_normalized = normalize(X, axis=0)
        normed_euc_items = find_nn(NearestNeighbors(n_neighbors=6), X_normalized.T, grill_brush_ind)

        print("Similar items using normalized Euclidean distance:")
        for i in normed_euc_items:
            print("  ", url_amazon % item_inverse_mapper[i])

        cos_items = find_nn(NearestNeighbors(n_neighbors=6, metric='cosine'), X.T, grill_brush_ind)

        print("Similar items using cosine similarity:")
        for i in cos_items:
            print("  ", url_amazon % item_inverse_mapper[i])

        print("Euclidean distance neighbours popularity", [np.sum(X_binary[:,i]) for i in euc_items])
        print("Cosine similarity  neighbours popularity", [np.sum(X_binary[:,i]) for i in cos_items])


    elif question == "2":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        ''' YOUR CODE HERE '''
        # Fit weighted least-squares estimator
        z = np.concatenate(([1]*400,[0.1]*100),axis = 0)
        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,z)

        utils.test_and_plot(model,X,y,title="Weighted Least Squares",filename="least_squares_outliers_weighted.pdf")

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        ''' YOUR CODE HERE'''
        # Fit the least squares model with bias
        model = linear_model.LeastSquaresBias()
        model.fit(X, y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, with bias",filename="least_squares_bias.pdf")

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p=%d" % p)

            ''' YOUR CODE HERE '''
            # Fit least-squares model
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X, y)

            utils.test_and_plot(model,X,y,Xtest,ytest,title='Least Squares Polynomial p = %d'%p,filename="PolyBasis%d.pdf"%p)
        
    else:
        print("Unknown question: %s" % question)
