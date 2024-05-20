import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def evaluate_model(model, features, labels):
    predictions = model.predict(features)
    rmse_score = rmse(labels, predictions)
    mae_score = mae(labels, predictions)
    return predictions, rmse_score, mae_score
