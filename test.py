import pandas as pd
import numpy as np
import polars as pl
from sklearn.preprocessing import OrdinalEncoder
import wandb
from wandb.lightgbm import log_summary
import lightgbm as lgb
from config import Cfg
from utils import MAE

def read_csv(fname="train.csv", path="./input"):
    return pl.read_csv(os.path.join(path, fname), try_parse_dates=True)

class PreProcessingPipeline:
    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers
    def __call__(self, df, client_df, revealed_target, gas_prices, electricity_prices, historical_weather, forecast_weather, locations):
        # データの前処理を実装する
        # ...


        return df_with_extra_cols.to_pandas()

cfg = Cfg()
wandb.init()

train = read_csv("train.csv", cfg.input_dir).with_columns(pl.col("county").cast(pl.datatypes.Int8))
client = read_csv("client.csv", cfg.input_dir).with_columns(pl.col("county").cast(pl.datatypes.Int8))
revealed_target_cols = ['prediction_unit_id', 'is_consumption', 'target', 'datetime']
revealed_target = (train
                   .select(revealed_target_cols)
                  )
gas_prices = read_csv("gas_prices.csv", cfg.input_dir)
electricity_prices = read_csv("electricity_prices.csv", cfg.input_dir)
historical_weather = read_csv("historical_weather.csv", cfg.input_dir)
future_weather = read_csv("forecast_weather.csv", cfg.input_dir)
locations = read_csv("county_lon_lats.csv", cfg.input_dir).drop("").with_columns(pl.col("county").cast(pl.datatypes.Int8))

# データ前処理パイプラインのインスタンス化
pp_pipeline = PreProcessingPipeline(list_of_transformers)

# データ前処理の実行
train_pd = pp_pipeline(train, client, revealed_target, gas_prices, electricity_prices, historical_weather, future_weather, locations)

mask = np.logical_and(~train_pd["target"].isna(), ~train_pd["revealed_target"].isna())
train_pd_non_null = train_pd[mask]

# 特徴量とターゲットの分割
X = train_pd.drop("target")
y = train_pd["target"]

# LightGBMモデルの学習と交差検証
params = {
    "objective": "regression",
    "metric": "mean_absolute_error",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1
}

scores = []
kf = KFold(n_splits=5, random_state=cfg.seed, shuffle=True)
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[train_data, val_data], early_stopping_rounds=100, verbose_eval=100)
    
    y_pred = model.predict(X_val)
    score = MAE(y_val, y_pred)
    scores.append(score)

mean_score = np.mean(scores)
print("cv", format(mean_score, ".5f"))
wandb.config["cv"] = mean_score
log_summary(model)
wandb.finish()