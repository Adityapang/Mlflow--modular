import pandas as pd
import hashlib
from typing import List, Tuple


def load_raw_data(path: str, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def get_data_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()



def temporal_train_test_split(df: pd.DataFrame,
                              date_col: str,
                              cutoff_week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_date = df[date_col].max() - pd.Timedelta(weeks=cutoff_week)
    train_df = df[df[date_col] <= cutoff_date].copy()
    test_df = df[df[date_col] > cutoff_date].copy()
    return train_df, test_df


def sort_data(train_df: pd.DataFrame,
              test_df: pd.DataFrame,
              date_col:str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.sort_values(date_col).reset_index(drop=True)
    test_df = test_df.sort_values(date_col).reset_index(drop=True)
    return train_df, test_df


def drop_null_values(train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                     target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_null_idx = train_df[train_df[target_col].isnull()].index
    train_df = train_df.drop(index = train_null_idx).reset_index(drop = True)

    test_null_idx = test_df[test_df[target_col].isnull()].index
    test_df = test_df.drop(index=test_null_idx).reset_index(drop=True)

    return train_df, test_df




def extract_data_metadata(df: pd.DataFrame, date_col: str, train: bool = False) -> dict:
    if train:
        return {
            "train_records": len(df),
            "train_date_min": df[date_col].min().strftime("%Y-%m-%d"),
            "train_date_max": df[date_col].max().strftime("%Y-%m-%d")
        }
    else:
        return {
            "test_records": len(df),
            "test_date_min": df[date_col].min().strftime("%Y-%m-%d"),
            "test_date_max": df[date_col].max().strftime("%Y-%m-%d")
        }


def split_features_target(df: pd.DataFrame,
                          target_col: str,
                          date_col: str,
                          return_meta:bool = False,
                          test_meta_cols: list = None)-> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col, date_col], errors="ignore")
    y = df[target_col]

    if return_meta:
        X_meta = df[test_meta_cols]
        return X, y, X_meta

    return X, y

def save_dataframe_to_csv(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)