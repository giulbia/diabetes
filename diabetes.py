import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import datasets
import argparse

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# Evaluate metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_diabetes():

    parser = argparse.ArgumentParser()

    parser.add_argument('alpha', type=float)
    parser.add_argument('l1_ratio', type=float)

    # Arguments
    args = parser.parse_args()

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # # Set default values
    # if float(alpha) is None:
    #     alpha = 0.05
    # else:
    #     alpha = float(alpha)
    #
    # if float(l1_ratio) is None:
    #     l1_ratio = 0.05
    # else:
    #     l1_ratio = float(l1_ratio)

    np.random.seed(40)

    # Load Diabetes datasets
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Create pandas DataFrame for sklearn ElasticNet linear_model
    Y = np.array([y]).transpose()
    d = np.concatenate((X, Y), axis=1)
    cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
    data = pd.DataFrame(d, columns=cols)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    train_x = train.drop(["progression"], axis=1)
    test_x = test.drop(["progression"], axis=1)
    train_y = train[["progression"]]
    test_y = test[["progression"]]
  
    # Start an MLflow run
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out ElasticNet model metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("RMSE: %s" % rmse)
        print("MAE: %s" % mae)
        print("R2: %s" % r2)
        
        # Log mlflow attributes for mlflow UI
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "model")


if __name__ == "__main__":
    train_diabetes()
