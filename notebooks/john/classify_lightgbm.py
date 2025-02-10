import optuna
import pandas as pd
#import lightgbm as lgb
from lightgbm import LGBMClassifier
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import os
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

def train_test_lightgbm(df_train: pd.DataFrame, 
                        df_test: pd.DataFrame, 
                        features: list[str],
                        remove_features: list[str],
                        target_col: str, 
                        apply_smote: bool = False,
                        cv: int = 5) -> tuple:

    # Convert target column to int8
    df_train[target_col] = df_train[target_col].astype('int32')
    df_test[target_col] = df_test[target_col].astype('int32')
    
        
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    
    if remove_features:
        features = [x for x in features if x not in remove_features]
    
    if apply_smote:
        # Increase training data using SMOTE
        print("Target before data augmentation", df_train[target_col].value_counts())
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(df_train[features], df_train[target_col])
        df_train = pd.concat([X_res, y_res], axis=1)
        print("Target after data augmentation", df_train[target_col].value_counts())
        
    print("Features:", features)
    
    n_opt_trials = 50
    params = {
        "objective": "binary",
        "metric": "f1_score",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_seed": random_seed,
    }

    # Split data into train and test
    train, val = train_test_split(df_train, test_size=0.1, random_state=random_seed)

    train_x = train[features]
    train_y = train[target_col]

    val_x = val[features]
    val_y = val[target_col]

    x_test= df_test[features]
    y_test = df_test[target_col]

    
    gbm = LGBMClassifier(**params)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
    y_pred = gbm.predict(val_x)
    f1 = f1_score(val_y, y_pred)

    print(f"F1 Score No Optimization: {f1}")
    
    # callable for optimization
    def objective(trial):
        # Parameters
        params_tuning = {
            "objective": "binary",
            "metric": "f1_score",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "feature_pre_filter": False,
        }

        gbm = LGBMClassifier(**params_tuning)
        gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
        y_pred = gbm.predict(val_x)
        return -f1_score(val_y, y_pred)

    # start hyperparameter tuning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_opt_trials)

    print("Best hyperparameters:", study.best_params)
    print("Best f1_score:", study.best_value*(-1))

    # Model Training with best params and performance test
    X_train = df_train[features]
    y_train = df_train[target_col]
    
    best_params = study.best_params
    best_params.update({"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "boosting_type": "gbdt"})

    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    # prediction
    y_pred = model.predict(x_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    f1_score_result = f1_score(y_test, y_pred)
    accuracy_score_result = accuracy_score(y_test, y_pred)
    
    print(f"Acurracy Score: {accuracy_score_result}")   
    print(f"Classification Report: {class_report}")
    
    cv_scores = None
    #cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(f"Cross Validation Scores: {cv_scores}")
    
    # Modell evaluieren
    train_score = model.score(X_train, y_train)
    test_score = model.score(x_test, y_test)
    
    scores = {
        "cv_scores": cv_scores,
        'train_score': train_score,
        'test_score': test_score
        
    }
    
    metrics = {
        "class_report": class_report,
        "conf_matrix": conf_matrix,
        "f1_score": f1_score_result,
        "accuracy_score": accuracy_score_result
    }
    
    return model, features, scores, metrics

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

    model, features, scores, metrics = train_test_lightgbm(df_train=df_train_step_2, 
                                df_test=df_test_step_2, 
                                features=features,
                                remove_features=['most_current_software_update'],
                                target_col=target_columns[0], 
                                apply_smote=False)

    models_metrics['step_2'] = {
        'model': model,
        'features': features,
        'scores': scores,
        'metrics': metrics
    }