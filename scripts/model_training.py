from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_random_forest(features, labels):
    param_grid = {
        'n_estimators': [25, 50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, scoring='neg_mean_squared_error', cv=5, verbose=1, random_state=42, n_jobs=-1)
    random_search.fit(features, labels)
    best_params = random_search.best_params_
    return best_params

def train_rf_model(features, labels, best_params):
    rf_model = RandomForestRegressor(**best_params)
    rf_model.fit(features, labels)
    return rf_model
