import diffprivlib as dp
import numpy as np


# takes a list of columns to which DP should be applied, a dataframe and values for delta, epsilon and sensitivity and applies the Laplace mechanism
# if clipping is set to true, negative numbers are rounded to zero
# if rounding is set to True, floats are rounded to the nearest integer
# by using random state, deterministic behaviour of DP can be achieved
def apply_laplace(
    df, columns, epsilon, delta, sensitivity, clipping, rounding, random_state=None
):
    laplace = dp.mechanisms.Laplace(
        epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state
    )
    df_dp = df.copy(deep=True)
    for col in columns:
        df_dp[col] = df[col].apply(laplace.randomise)
        if clipping:
            df_dp[col] = np.clip(df_dp[col], 0, None)
        if rounding:
            df_dp[col] = np.round(df_dp[col].astype(int))
    return df_dp


# applies DP to a row and takes a seed as random value
def laplace_seed(x, seed, epsilon, delta, sensitivity, clipping, rounding):
    laplace = dp.mechanisms.Laplace(
        epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=seed
    )
    x_dp = x.apply(laplace.randomise)
    if clipping:
        x_dp = np.clip(x_dp, 0, None)
    if rounding:
        x_dp = np.round(x_dp.astype(int))
    return x_dp


# creates a new column with differentially private data in an existing data frame
# if clipping is set to true, negative numbers are rounded to zero
# if rounding is set to True, floats are rounded to the nearest integer
# by using random state, deterministic behaviour of DP can be achieved
def create_dp_column_laplace(
    df,
    column,
    column_name,
    epsilon,
    delta,
    sensitivity,
    clipping,
    rounding,
    random_state=None,
):
    laplace = dp.mechanisms.Laplace(
        epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state
    )
    df[column_name] = df[column].apply(laplace.randomise)
    if clipping:
        df[column_name] = np.clip(df[column_name], 0, None)
    if rounding:
        df[column_name] = np.round(df[column_name].astype(int))
    return df


# applies DP to a row and takes a seed as random value
def geometric_seed(x, seed, epsilon, sensitivity, clipping):
    geometric = dp.mechanisms.Geometric(
        epsilon=epsilon, sensitivity=sensitivity, random_state=seed
    )
    print(seed)
    x_dp = x.apply(geometric.randomise)

    if clipping:
        x_dp = np.clip(x_dp, 0, None)
    return x_dp


# takes a list of columns to which DP should be applied, a dataframe and values for delta, epsilon and sensitivity and applies the Geometric mechanism
# if clipping is set to True, negative numbers are rounded to zero
# rounding is not needed here since the geometric mechanism returns integers
# by using random state, deterministic behaviour of DP can be achieved
def apply_geometric(df, columns, epsilon, sensitivity, clipping, random_state=None):
    geometric = dp.mechanisms.Geometric(
        epsilon=epsilon, sensitivity=sensitivity, random_state=random_state
    )
    df_dp = df.copy(deep=True)
    for col in columns:
        df_dp[col] = df[col].apply(geometric.randomise)
        if clipping:
            df_dp[col] = np.clip(df_dp[col], 0, None)
    return df_dp


# creates a new column with differentially private data in an existing data frame
# if clipping is set to true, negative numbers are rounded to zero
# rounding is not needed here since the geometric mechanism returns integers
# by using random state, deterministic behaviour of DP can be achieved
def create_dp_column_geometric(
    df, column, column_name, epsilon, sensitivity, clipping, random_state=None
):
    geometric = dp.mechanisms.Geometric(
        epsilon=epsilon, sensitivity=sensitivity, random_state=random_state
    )
    df[column_name] = df[column].apply(geometric.randomise)
    if clipping:
        df[column_name] = np.clip(df[column_name], 0, None)
    return df
