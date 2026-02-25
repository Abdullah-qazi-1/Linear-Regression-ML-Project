import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.utils import save_model
from src.exception import CustomException
from src.logger import logging


def load_data(path):
    try:
        logging.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        logging.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data. Error in file [{__file__}] at line [{e.__traceback__.tb_lineno}]: {str(e)}")    
        raise CustomException(e)


def prepare_data(df, target_column):
    try:
        logging.info(f"Preparing data by separating features and target column '{target_column}'")
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        logging.info(f"Data preparation complete. Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    except Exception as e:
        logging.error(f"Error  preparing data. Error in file [{__file__}] at line [{e.__traceback__.tb_lineno}]: {str(e)}")
        raise CustomException(e)


def split_data(X, y, test_size=0.2):
    try:
        logging.info(f"Splitting data into train and test sets with test size {test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        logging.info(f"Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data. Error in file [{__file__}] at line [{e.__traceback__.tb_lineno}]: {str(e)}")
        raise CustomException(e)

def train_model(X_train, y_train, model_type='linear'):
    try:
        logging.info(f"Training model of type '{model_type}'")
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'decision_tree':
            model = DecisionTreeRegressor(random_state=42)
            logging.info(f"Model '{model_type}' initialized successfully")  
        else:
            raise ValueError("Unsupported model type")
        logging.info(f"Fitting the model to the training data")
        
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error(f"Error training model. Error in file [{__file__}] at line [{e.__traceback__.tb_lineno}]: {str(e)}")
        raise CustomException(e)
    
def evaluate_model(model, X_test, y_test):
    try:
        logging.info(f"Evaluating model performance on test data")
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"=== {type(model).__name__} ===")
        print(f"R2 Score : {r2:.4f}")
        print(f"MAE      : {mae:.4f}")
        logging.info(f"Model evaluation complete. R2 Score: {r2}, MAE: {mae}")
        return r2, mae
    except Exception as e:
        logging.error(f"Error evaluating model. Error in file [{__file__}] at line [{e.__traceback__.tb_lineno}]: {str(e)}")
        raise CustomException(e)