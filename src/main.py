import logging
from os import getenv
from pathlib import Path
from time import perf_counter

import pandas as pd
from catboost import CatBoostRegressor, Pool
from dotenv import load_dotenv

import wandb

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

load_dotenv(config_path / '.env')
wandb.login(host='https://api.wandb.ai', key=getenv('WANDB_KEY'))

df_train = pd.read_parquet(dataset_path / 'green_tripdata_2021-01.parquet')
df_val = pd.read_parquet(dataset_path / 'green_tripdata_2021-02.parquet')

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
    # 'controller': {'type': 'local'},
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'best_score_val'},
    'parameters':
        {
            'learning_rate': {'max': 0.13, 'min': 0.01},
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

    with wandb.init(project='taxi',
                    # dir=wandb_path,
                    config={'train_dir': catboost_path,
                            'objective': 'RMSE',
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
                   'best_score_train': best_score['learn']['RMSE'],
                   'best_score_val': best_score['validation']['RMSE'],
                   'training_time': elapsed_time})


wandb.agent(sweep_id, function=train, count=10)
# sweep = wandb.controller(sweep_id)
# sweep.run(verbose=True, print_actions=True)
