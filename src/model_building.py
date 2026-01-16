import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict
) -> RandomForestRegressor:
    """
    Train the RandomForest Regressor model using provided hyperparameters.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples")

        logger.debug("Initializing RandomForestRegressor with params: %s", params)

        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt",
            max_samples=params["max_samples"],
            bootstrap=params["bootstrap"],
            n_jobs=-1,
            random_state=42
        )

        logger.debug("Training started with %d samples", X_train.shape[0])
        model.fit(X_train, y_train)
        logger.debug("Training completed")

        return model

    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise



def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # 🔧 Hyperparameters controlled from here
        params = load_params('params.yaml')['model_building']

        train_data = load_data("./data/processed/train_engineered.csv")

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        model = train_model(X_train, y_train, params)

        save_model(model, "models/model.pkl")

        logger.info("RandomForestRegressor training completed successfully")

    except Exception as e:
        logger.error("Model training failed: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
