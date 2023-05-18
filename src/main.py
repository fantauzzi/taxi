import logging
from os import getenv
from pathlib import Path
from time import perf_counter
from urllib.request import urlretrieve

import pandas as pd
import wandb
from catboost import CatBoostRegressor, Pool
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
_logger = logging.getLogger()
info = _logger.info
warning = _logger.warning

prj_root = Path('..').resolve()
dataset_path = (prj_root / 'dataset').resolve()
config_path = (prj_root / 'config').resolve()
# wandb_path = (prj_root / 'wandb').resolve()
catboost_path = (prj_root / 'catboost_info').resolve()
pd.set_option('display.max_columns', None)
train_file = 'green_tripdata_2021-01.parquet'
val_file = 'green_tripdata_2021-02.parquet'
train_file_remote = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet'
val_file_remote = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet'
dot_env_path = config_path / '.env'

if Path(dot_env_path).exists():
    load_dotenv(dot_env_path)
if getenv('WANDB_KEY') is None:
    info(f'WANDB_KEY is not set. Now trying to log into Weights & Biases; if unable, either set the environment \
variable WANDB_KEY to the key to be used, or set it in {dot_env_path}')
wandb.login(host='https://api.wandb.ai', key=getenv('WANDB_KEY'))

train_path = dataset_path / train_file
if not Path(train_path).exists():
    info(f'Downloading dataset {train_file_remote} into {train_path}')
    urlretrieve(train_file_remote, train_path)

val_path = dataset_path / val_file
if not Path(val_path).exists():
    info(f'Downloading dataset {val_file_remote} into {val_path}')
    urlretrieve(val_file_remote, val_path)

df_train = pd.read_parquet(train_path)
df_val = pd.read_parquet(val_path)

cat_features = ['PULocationID', 'DOLocationID']

numerical = ['trip_distance']


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(['store_and_fwd_flag'], axis=1)
    df.loc[:, 'duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df = df.drop(['lpep_pickup_datetime', 'lpep_dropoff_datetime'], axis=1)
    df.loc[:, cat_features] = df[cat_features].astype(str)
    # df = df.loc[:, cat_features + numerical+['duration']]
    return df


df_train = prepare(df_train)
# df_train = df_train[:20]
df_val = prepare(df_val)

info(f'Training set contains {len(df_train)} samples')
info(f'Validation set contains {len(df_val)} samples')

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'best_score_val'},
    'parameters':
        {
            'learning_rate': {'max': 0.3, 'min': 0.01},
            'depth': {'max': 10, 'min': 6},
            'border_count': {'value': 128}
        }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='taxi_sweep')


def train():
    start_time = perf_counter()
    train_pool = Pool(df_train.drop('duration', axis=1),
                      label=df_train['duration'],
                      cat_features=cat_features)
    val_pool = Pool(df_val.drop('duration', axis=1),
                    label=df_val['duration'],
                    cat_features=cat_features)
    """
    test_pool = Pool(df_test.drop('duration', axis=1),
                      label = df_test['duration'],
                      cat_features=cat_features)"""

    iterations = 20000
    early_stopping_rounds = 200
    objective = 'MAPE'
    with wandb.init(project='taxi',
                    # dir=wandb_path,
                    config={'train_dir': catboost_path,
                            'objective': objective,
                            'task_type': 'GPU',
                            'iterations': iterations,
                            'early_stopping_rounds': early_stopping_rounds,
                            'per_float_feature_quantization': '5:border_count=1024'}) as run:
        model = CatBoostRegressor(**wandb.config)

        verbose = 200
        model.fit(train_pool,
                  eval_set=val_pool,
                  use_best_model=True,
                  verbose=verbose)

        best_score = model.get_best_score()
        elapsed_time = perf_counter() - start_time
        print(f'Training time (sec) {elapsed_time}')
        wandb.log({'best_iteration': model.get_best_iteration(),
                   'best_score_train': best_score['learn'][objective],
                   'best_score_val': best_score['validation'][objective],
                   'training_time': elapsed_time})


wandb.agent(sweep_id, function=train, count=20)

""" 
TODO: 
set seed/reproducibility
add artifical variable with day of the week
"""
