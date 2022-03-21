from typing import List
import pandas as pd


def concat_dataframes(dataframes: List[pd.DataFrame], idx_column_name: str) -> pd.DataFrame:
    """
    Concatenates dataframes as a MultiIndex Dataframe
    :param dataframes: List of dataframes
    :param idx_column_name: Name of the new index column
    :return:
    """
    df = pd.concat(dataframes, keys=[i for i in range(len(dataframes))], names=[idx_column_name, ])
    return df
