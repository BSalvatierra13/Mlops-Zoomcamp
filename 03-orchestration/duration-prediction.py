import pandas as pd
import pickle

import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import xgboost as xgb

from pathlib import Path



mlflow.set_tracking_uri("sqlite:///../mlflow.db")
experiment_name = "nyc-taxi-orches"


def read_dataframe(year, month):
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

df_train = read_dataframe(2021, 1)
df_val = read_dataframe(2021, 2)

def create_X(df, dv=None):
    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv





# In[38]:
def train_model(X_train, y_train, X_val, y_val, dv):
        
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("../models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("../models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

def run(year, month):
    df_train = read_dataframe(year, month)

    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    df_val = read_dataframe(next_year, next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    train_model(X_train, y_train, X_val, y_val, dv)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default=2021, type=int)
    parser.add_argument("--month", default=1, type=int)
    args = parser.parse_args()
    run(args.year, args.month)

    #save run.id
    run_id = mlflow.active_run().info.run_id
    with open("../models/run_id.txt", "w") as f_out:
        f_out.write(run_id)


