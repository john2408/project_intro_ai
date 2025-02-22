from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import (
    FTTransformerConfig,
    GatedAdditiveTreeEnsembleConfig
)
from pytorch_tabular.models.common.heads import LinearHeadConfig
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, log_loss

from omegaconf import DictConfig
import pickle
import optuna
torch.serialization.safe_globals([DictConfig])

# Adjust Function pytorch_tabular/utils/python_utils.py

# def pl_load(
#     path_or_url: Union[IO, _PATH],
#     map_location: _MAP_LOCATION_TYPE = None,
# ) -> Any:
#     """Loads a checkpoint.

#     Args:
#         path_or_url: Path or URL of the checkpoint.
#         map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.

#     """
#     if not isinstance(path_or_url, (str, Path)):
#         # any sort of BytesIO or similar
#         return torch.load(path_or_url, map_location=map_location, weights_only=False)
#     if str(path_or_url).startswith("http"):
#         return torch.hub.load_state_dict_from_url(
#             str(path_or_url),
#             map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
#         )
#     fs = get_filesystem(path_or_url)
#     with fs.open(path_or_url, "rb") as f:
#         return torch.load(f, map_location=map_location, weights_only=False)


def train_test_ft_tabular(df_train: pd.DataFrame, 
                          df_test: pd.DataFrame, 
                          num_cols: list[str], 
                          cat_cols: list[str],
                          target_col: list[str],
                          n_opt_trials: int = 10,
                          random_seed: int = 123) -> tuple:
   
    train, val = train_test_split(df_train, test_size=0.2, random_state=random_seed)
    
    val_x = val.drop(columns=['user_of_latest_model'])  # Features
    val_y = val['user_of_latest_model']  # Target variable
    
    X_test = df_test.drop(columns=['user_of_latest_model'])  # Features
    y_test = df_test['user_of_latest_model']  # Target variable

    data_config = DataConfig(
            target=target_col,  # target should always be a list.
            continuous_cols=num_cols,
            categorical_cols=cat_cols,
    )
    
    def objective(trial):
        
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16]) 
        num_attn_blocks = trial.suggest_int("num_attn_blocks", 1, 4)
        
        
        trainer_config = TrainerConfig(
            #     auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
            batch_size=32,
            max_epochs=30,
            early_stopping="valid_loss",  # Monitor valid_loss for early stopping
            early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
            early_stopping_patience=5,  # No. of epochs of degradation training will wait before terminating
            checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
                load_best=True,  # After training, load the best checkpoint
        )

        optimizer_config = OptimizerConfig()

        head_config = LinearHeadConfig(
            layers="",  # No additional layer in head, just a mapping layer to output_dim
            dropout=0.1,
            initialization="kaiming",
        ).__dict__ 
            
        
        # FT Tabular Transformer
        model_config = FTTransformerConfig(
            task="classification",
            head="LinearHead",  # Linear Head
            head_config=head_config,  # Linear Head Config
            learning_rate = learning_rate,
            num_heads=num_heads,
            num_attn_blocks=num_attn_blocks,
        )
        
        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        tabular_model.fit(train=train, validation=val)
        
        y_pred_val = np.array(tabular_model.predict(test=val_x))[:, -1]
        val_f1_score = round(f1_score(val_y, y_pred_val),4)
            
        return -val_f1_score
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_opt_trials)
    best_params = study.best_params

    trainer_config = TrainerConfig(
        #     auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
        batch_size=32,
        max_epochs=30,
        early_stopping="valid_loss",  # Monitor valid_loss for early stopping
        early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
        early_stopping_patience=5,  # No. of epochs of degradation training will wait before terminating
        checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
        load_best=True,  # After training, load the best checkpoint
    )

    optimizer_config = OptimizerConfig()

    head_config = LinearHeadConfig(
        layers="",  # No additional layer in head, just a mapping layer to output_dim
        dropout=0.1,
        initialization="kaiming",
    ).__dict__ 

    # FT Tabular Transformer
    model_config = FTTransformerConfig(
        task="classification",
        head="LinearHead",  # Linear Head
        head_config=head_config,  # Linear Head Config
        learning_rate=best_params['learning_rate'],
        num_heads=best_params['num_heads'],
        num_attn_blocks=best_params['num_attn_blocks'],
    )
    
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    
    tabular_model.fit(train=train, validation=val)
    
    y_pred_val = np.array(tabular_model.predict(test=val_x))[:, -1]
    val_accuracy = round(accuracy_score(val_y, y_pred_val),4)
    val_f1_score = round(f1_score(val_y, y_pred_val),4)
    
    y_pred = np.array(tabular_model.predict(test=X_test))[:, -1]  
    test_accuracy = round(accuracy_score(y_test, y_pred),4)
    test_f1_score = round(f1_score(y_test, y_pred),4)
    model_name = "FTTabular"
    
    df_results = pd.DataFrame({'Model_Name': [model_name], 
                               'val_f1_score': [val_f1_score], 
                               'val_accuracy': [val_accuracy], 
                               'test_f1_score': [test_f1_score],
                               'test_accuracy': [test_accuracy]})
  
    return df_results


if __name__ == "__main__":

    target_columns = ['user_of_latest_model']

    models_metrics = {} 

    # Store the dataframe to use them in the next step
    df_train = pd.read_csv(os.path.join(os.getcwd(),"data/processed/task2_best_model_step3_train_data.csv"))
    df_test = pd.read_csv(os.path.join(os.getcwd(),"data/processed/task2_best_model_step3_test_data.csv"))


    # OneHot-Encoding for categorical columns
    df_train = pd.get_dummies(df_train, columns=df_train.select_dtypes(include=['object']).columns.to_list(), prefix="_cat", drop_first=True)
    df_test = pd.get_dummies(df_test, columns=df_test.select_dtypes(include=['object']).columns.to_list(), prefix="_cat", drop_first=True)

    features = df_train.columns.to_list()
    features.remove('user_of_latest_model')
    cat_cols = [col for col in features if "_cat" in col]
    num_cols = [col for col in features if "_cat" not in col]

    df_results = train_test_ft_tabular(df_train=df_train, 
                                df_test=df_test, 
                                num_cols=num_cols, 
                                cat_cols=cat_cols,
                                target_col=target_columns)
    
    print(df_results)
    
    # Store results as pkl
    with open(os.path.join(os.getcwd(), "data/processed/ftt_model_results_optuna.pkl"), "wb") as f:
        pickle.dump(df_results, f)
