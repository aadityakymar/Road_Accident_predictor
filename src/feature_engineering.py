import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml
from sklearn.preprocessing import StandardScaler
import numpy as np
# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.debug('Parameters retrieved from %s', params_path)
#         return params
#     except FileNotFoundError:
#         logger.error('File not found: %s', params_path)
#         raise
#     except yaml.YAMLError as e:
#         logger.error('YAML error: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error: %s', e)
#         raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-driven features for road accident analysis.
    """
    try:
        df = df.copy()

        # -------- Road Geometry --------
        df["high_curvature"] = (df["curvature"] > df["curvature"].quantile(0.75)).astype(int)
        df["road_complexity"] = df["num_lanes"] * df["curvature"]

        # -------- Speed & Regulation --------
        df["high_speed_zone"] = (df["speed_limit"] >= 80).astype(int)
        df["speed_lane_ratio"] = df["speed_limit"] / (df["num_lanes"] + 1)

        # -------- Traffic Exposure --------
        df["accident_density"] = df["num_reported_accidents"] / (df["num_lanes"] + 1)
        

        logger.debug("Feature engineering completed")
        return df

    except Exception as e:
        logger.error("Error during feature engineering: %s", e)
        raise

def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Normalize newly created numerical features only.
    """
    try:
        engineered_cols = [
            "road_complexity",
            "speed_lane_ratio",
            "accident_density"
        ]

        scaler = StandardScaler()

        train_df[engineered_cols] = scaler.fit_transform(train_df[engineered_cols])
        test_df[engineered_cols] = scaler.transform(test_df[engineered_cols])

        logger.debug("Engineered features normalized")
        return train_df, test_df

    except Exception as e:
        logger.error("Error during feature normalization: %s", e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        train_data = load_data("./data/interim/train_processed.csv")
        test_data = load_data("./data/interim/test_processed.csv")

        train_fe = engineer_features(train_data)
        test_fe = engineer_features(test_data)

        train_fe, test_fe = normalize_features(train_fe, test_fe)

        save_data(train_fe, "./data/processed/train_engineered.csv")
        save_data(test_fe, "./data/processed/test_engineered.csv")

        logger.info("Feature engineering pipeline completed successfully")

    except Exception as e:
        logger.error("Feature engineering pipeline failed: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
