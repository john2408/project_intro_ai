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
                          remove_features: list[str] = None) -> tuple:
   
    
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    
    if remove_features:
        num_cols = [x for x in num_cols if x not in remove_features]
        cat_cols = [x for x in cat_cols if x not in remove_features]
    
        # Convert target column to int8
    df_train[target_col] = df_train[target_col].astype('int32')
    df_test[target_col] = df_test[target_col].astype('int32')
    
    # Split data into train and test
    train, val = train_test_split(df_train, test_size=0.1, random_state=random_seed)

    x_test= df_test[features]
    y_test = df_test[target_col].values

    # Config 
    data_config = DataConfig(
        target=target_col,  # target should always be a list.
        continuous_cols=num_cols,
        categorical_cols=cat_cols,
    )

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
        learning_rate=1e-3,
        head="LinearHead",  # Linear Head
        head_config=head_config,  # Linear Head Config
    )
    
    # model_config = GatedAdditiveTreeEnsembleConfig(
    # task="classification",
    # learning_rate=1e-3,
    # head="LinearHead",  # Linear Head
    # head_config=head_config,  # Linear Head Config
    # )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    
        
    tabular_model.fit(train=train, validation=val)
    y_pred= np.array(tabular_model.predict(test=x_test))[:, -1]  
    
      
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    f1_score_result = f1_score(y_test, y_pred)
    accuracy_score_result = accuracy_score(y_test, y_pred)
    
    print("F1 score: ", f1_score_result)
    print(f"Acurracy Score: {accuracy_score_result}")   
    print(f"Classification Report: {class_report}")
    
    cv_scores = None
    #cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(f"Cross Validation Scores: {cv_scores}")
    
    scores = None
    
    metrics = {
        "class_report": class_report,
        "conf_matrix": conf_matrix,
        "f1_score": f1_score_result,
        "accuracy_score": accuracy_score_result
    }
  
    return tabular_model, eval_metrics



if __name__ == "__main__":

    category_columns = ['sex',
    'ctry',
    'town',
    'goal_of_training',
    'preferred_training_daytime',
    'subscription_type',
    'color_of_watch']

    text_columns = ['id']

    target_columns = ['user_of_latest_model']

    models_metrics = {} 

    
    test_data_path = os.path.join(os.getcwd(), "data/processed/triathlon_watch_test_preprocessed_john.csv")
    df_test_step_2 = pd.read_csv(test_data_path)

    # Add hot encoded features
    df_hot_encoded_feat = pd.get_dummies(df_test_step_2.filter(category_columns), 
                                        columns=category_columns, drop_first=True, 
                                        prefix="hot_enc_", dtype=int)
    df_test_step_2 = pd.concat([df_test_step_2, df_hot_encoded_feat], axis=1)
    df_test_step_2 = df_test_step_2.drop(category_columns + text_columns, axis=1)

    train_data_path = os.path.join(os.getcwd(),"data/processed/triathlon_watch_training_preprocessed_john_prep_2.csv")
    df_train_step_2 = pd.read_csv(train_data_path)


    # Verify dataframe have the same number of columns
    assert df_train_step_2.shape[1] == df_test_step_2.shape[1], "Columns mismatch between train and test dataset"

    features = [col for col in list(df_train_step_2.columns) if col != target_columns[0]]

    cat_cols = list(set(features) & set(category_columns))
    num_cols = list(set(features) - set(cat_cols))

    tabular_model, eval_metrics = train_test_ft_tabular(df_train=df_train_step_2, 
                                df_test=df_test_step_2, 
                                num_cols=num_cols, 
                                cat_cols=cat_cols,
                                target_col=target_columns,
                                remove_features=['most_current_software_update'],)
