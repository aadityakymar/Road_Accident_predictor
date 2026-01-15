import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
from sklearn.preprocessing import StandardScaler
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

target_column = "accident_risk"


CATEGORICAL_COLS = [
    "road_type",
    "lighting",
    "weather",
    "time_of_day"
]

BOOLEAN_COLS = [
    "road_signs_present",
    "public_road",
    "holiday",
    "school_season"
]

NUMERICAL_COLS = [
    "num_lanes",
    "curvature",
    "speed_limit",
    "num_reported_accidents"
]


def encode_boolean(df, bool_cols):
    for col in bool_cols:
        df[col] = df[col].astype(int)
    return df


def one_hot_encode(train_df, test_df, categorical_cols):
    train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

    # Align columns (CRUCIAL)
    train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

    return train_df, test_df


def normalize_features(train_df, test_df, num_cols):
    scaler = StandardScaler()

    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    return train_df, test_df

def preprocess_data(train_df, test_df, target_column):
    try:
        logger.debug("Starting preprocessing pipeline")

        # Drop ID column
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        # Separate target
        y_train = train_df[target_column]
        y_test = test_df[target_column]

        X_train = train_df.drop(columns=[target_column])
        X_test = test_df.drop(columns=[target_column])

        # Boolean encoding
        X_train = encode_boolean(X_train, BOOLEAN_COLS)
        X_test = encode_boolean(X_test, BOOLEAN_COLS)
        logger.debug("Boolean columns encoded")

        # One-hot encoding
        X_train, X_test = one_hot_encode(X_train, X_test, CATEGORICAL_COLS)
        logger.debug("Categorical columns one-hot encoded")

        # Normalization
        X_train, X_test = normalize_features(X_train, X_test, NUMERICAL_COLS)
        logger.debug("Numerical features normalized")

        # Reattach target
        train_processed = pd.concat([X_train, y_train], axis=1)
        test_processed = pd.concat([X_test, y_test], axis=1)

        return train_processed, test_processed

    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        raise


def main():
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Raw data loaded")

        train_processed, test_processed = preprocess_data(
            train_data, test_data, target_column="accident_risk"
        )

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug("Processed data saved successfully")

    except Exception as e:
        logger.error("Pipeline execution failed: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
