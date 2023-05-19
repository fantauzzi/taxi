import logging
from functools import partial
from os import getenv, system
from pathlib import Path
from time import perf_counter
from urllib.request import urlretrieve

import hydra
import pandas as pd
import wandb
from catboost import CatBoostRegressor, Pool
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Configure Python logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
_logger = logging.getLogger()
info = _logger.info
warning = _logger.warning

pd.set_option('display.max_columns', None)

# Set various paths and filenames
prj_root = Path('..').resolve()
dataset_path = (prj_root / 'dataset').resolve()
config_path = (prj_root / 'config').resolve()
# wandb_path = (prj_root / 'wandb').resolve()
catboost_path = (prj_root / 'catboost_info').resolve()
train_file = 'green_tripdata_2021-01.parquet'
val_file = 'green_tripdata_2021-02.parquet'
train_file_remote = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet'
val_file_remote = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet'
dot_env_path = config_path / '.env'
sweep_config_path = config_path / 'sweep_config.yaml'

if Path(dot_env_path).exists():
    load_dotenv(dot_env_path)

if getenv('WANDB_KEY') is None:
    info(f'WANDB_KEY is not set. Now trying to log into Weights & Biases; if unable, either set the environment \
variable WANDB_KEY to the key to be used, set it in {dot_env_path}, or jost login with "wandb login"')
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

cat_features = ['PULocationID', 'DOLocationID', 'weekday']  # weekday is an artificial variable to be introduced below

numerical = ['trip_distance']


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(['store_and_fwd_flag'], axis=1)
    # Make a duration column with the difference in minutes between lpep_dropoff_datetime and lpep_pickup_datetime
    df.loc[:, 'duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    # Add an artificial variable with the day of the week at the time the ride started, lpep_pickup_datetime
    weekday = df['lpep_pickup_datetime'].apply(lambda entry: entry.weekday())
    df.loc[:, 'weekday'] = weekday
    # Drop columns lpep_pickup_datetime and lpep_dropoff_datetime,
    # otherwise they would leak the information to be predicted
    df = df.drop(['lpep_pickup_datetime', 'lpep_dropoff_datetime'], axis=1)

    df.loc[:, cat_features] = df[cat_features].astype(str)
    # df = df.loc[:, cat_features + numerical+['duration']]
    return df


df_train = prepare(df_train)
# df_train = df_train[:20]
df_val = prepare(df_val)

info(f'Training set contains {len(df_train)} samples')
info(f'Validation set contains {len(df_val)} samples')


def train(params: DictConfig) -> None:
    start_time = perf_counter()
    info(f'params is {params}')
    train_pool = Pool(df_train.drop('duration', axis=1),
                      label=df_train['duration'],
                      cat_features=cat_features)
    val_pool = Pool(df_val.drop('duration', axis=1),
                    label=df_val['duration'],
                    cat_features=cat_features)
    """ test_pool = Pool(df_test.drop('duration', axis=1),
                      label = df_test['duration'],
                      cat_features=cat_features) """
    objective = 'MAPE'

    ''' 
    The config passed to wandb.init() contains:
    - all parameters read by hydra from the YAML configuration file (possibly overridden by command line), converted
      to a dict and under the key 'params'
    - all parameters to be passed to CatBoostRegressor(), as key-value pairs
    '''
    with wandb.init(project=params.wandb.project,
                    # dir=wandb_path,
                    config={'params': OmegaConf.to_object(params),
                            'train_dir': catboost_path,
                            'objective': objective,
                            'task_type': params.catboost.task_type,
                            'random_seed': params.catboost.random_seed,
                            'iterations': params.catboost.iterations,
                            'early_stopping_rounds': params.catboost.early_stopping_rounds,
                            'per_float_feature_quantization': '5:border_count=1024'}) as run:
        # Extract from wandb.config the parameters to be passed to CatBoostRegressor()
        catboost_params = dict(wandb.config)
        del (catboost_params['params'])

        # Instantiate and fit the model
        model = CatBoostRegressor(**catboost_params)
        model.fit(train_pool,
                  eval_set=val_pool,
                  use_best_model=True,
                  verbose=params.catboost.verbose)

        # Log the resulting metrics
        best_score = model.get_best_score()
        elapsed_time = perf_counter() - start_time
        print(f'Training time (sec) {elapsed_time}')
        wandb.log({'best_iteration': model.get_best_iteration(),
                   'best_score_train': best_score['learn'][objective],
                   'best_score_val': best_score['validation'][objective],
                   'training_time': elapsed_time})


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    info(f'These are the parameter(s) set at the beginning of the sweep:')
    for key, value in params.items():
        info(f"  '{key}': {value}")

    # Configure the sweep
    info(f'Loading sweep configuration from {sweep_config_path}')
    sweep_configuration = OmegaConf.load(sweep_config_path)
    sweep_configuration = OmegaConf.to_object(sweep_configuration)

    # Start a sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=params.wandb.project)
    info(f'Starting sweep for {params.wandb.count} iteration(s) with id {sweep_id}')
    train_with_params = partial(train, params)
    wandb.agent(sweep_id, function=train_with_params, count=params.wandb.count)

    # Stop the sweep
    api = wandb.Api()
    prj = api.project(params.wandb.project)
    sweep_long_id = f'{prj.entity}/{params.wandb.project}/{sweep_id}'
    command = f'wandb sweep --stop {sweep_long_id}'
    info(f'Stopping the current sweep {sweep_id} with command:')
    info(f'  {command}')
    system(command)


if __name__ == '__main__':
    main()

"""
TODO: 

"""
