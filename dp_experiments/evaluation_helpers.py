import numpy as np
import diff_priv
import data_error
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import os
from copy import deepcopy
import random


# mu-smoothed KL-divergence
def mu_smoothed_kl_divergence(p, q, mu):
    return sum((p[i] + mu) * np.log((p[i] + mu) / (q[i] + mu)) for i in range(len(p)))


# uses num_minorities to classify diversity
def classify_diversity(number_of_minorities):
    if number_of_minorities <= 5:
        return "low"
    elif number_of_minorities <= 10:
        return "medium"
    else:
        return "high"


# this function counts number of ethnic minority groups (populations who make up over 0 and under 10% of the population)
def count_minorities(df, column_name):
    num_minorities = len(df.loc[(df[column_name] > 0) & (df[column_name] <= 0.1)])
    return num_minorities


# this function returns the n largest minority groups. Here the white ethnicity and populations which make out more than 10% of the population in the ward are excluded.
def largest_minority_groups(df, n, column_name):
    # filtering out the White population
    filtered_df = df[
        (df["EthnicGroup"] != "White: English/Welsh/Scottish/Northern Irish/British")
    ]
    # filtering out all non-minorities which make out more than 10% of population
    filtered_df = filtered_df.loc[(df[column_name] > 0) & (df[column_name] <= 10)]
    index = filtered_df[column_name].astype(float).nlargest(n).index
    filtered_df = filtered_df.loc[index]
    ethnicities = filtered_df["EthnicGroup"].tolist()
    return ethnicities


# this function creates a dictionary of wards and calculates some additional data like a diversity index and what the largest ethnic groups in this ward are
def get_filtered_df_ward_dict(
    process_bulk, ward, df_ward, ward_codes, lookup_df, filter_dict
):
    wards = {}
    for code in ward_codes:
        df, area_name = process_bulk.get_filtered_df_ward(
            ward, df_ward, code, lookup_df, filter_dict
        )
        total = df["PopulationNumbers"].sum()
        df["%"] = df["PopulationNumbers"] / total

        # if there are several subcategories, e.g. tenure etc, the percentages of minorities have to be added up
        df["total %"] = df.groupby("EthnicGroup")["%"].transform("sum")
        df_totals = df.drop_duplicates(subset=["EthnicGroup"])

        number_minorities = count_minorities(df_totals, "total %")
        number_ethnicities = len(df_totals.loc[(df_totals["total %"] > 0)])
        largest_groups = largest_minority_groups(df_totals, 3, "total %")
        wards[code] = [
            df,
            {
                "area_name": area_name,
                "total": total,
                "number_minorities": number_minorities,
                "largest_groups": largest_groups,
                "number_ethnicities": number_ethnicities,
            },
        ]

    return wards


def get_filtered_df_la_dict(process_bulk, la, df_la, la_codes, lookup_df, filter_dict):
    las = {}
    for code in la_codes:
        df, area_name = process_bulk.get_filtered_df_la(
            la, df_la, code, lookup_df, filter_dict
        )

        total = df["PopulationNumbers"].sum()
        df["%"] = df["PopulationNumbers"] / total

        # if there are several subcategories, e.g. tenure etc, the percentages of minorities have to be added up
        df["total %"] = df.groupby("EthnicGroup")["%"].transform("sum")
        df_totals = df.drop_duplicates(subset=["EthnicGroup"])

        number_minorities = count_minorities(df_totals, "total %")
        number_ethnicities = len(df_totals.loc[(df_totals["total %"] > 0)])
        largest_groups = largest_minority_groups(df_totals, 3, "total %")
        la[code] = [
            df,
            {
                "area_name": area_name,
                "total": total,
                "number_minorities": number_minorities,
                "largest_groups": largest_groups,
                "number_ethnicities": number_ethnicities,
            },
        ]

    return las


# this function creates a dictionary of las and calculates some additional data like a diversity index and what the largest ethnic groups in this las are
def get_filtered_df_la_dict(process_bulk, la, df_la, la_codes, lookup_df, filter_dict):
    las = {}
    for code in la_codes:
        df, area_name = process_bulk.get_filtered_df_la(
            la, df_la, code, lookup_df, filter_dict
        )

        total = df["PopulationNumbers"].sum()
        df["%"] = df["PopulationNumbers"] / total

        df["total %"] = df.groupby("EthnicGroup")["%"].transform("sum")
        df_totals = df.drop_duplicates(subset=["EthnicGroup"])

        number_minorities = count_minorities(df_totals, "total %")
        number_ethnicities = len(df_totals.loc[(df_totals["total %"] > 0)])
        largest_groups = largest_minority_groups(df_totals, 3, "total %")

        las[code] = [
            df,
            {
                "area_name": area_name,
                "total": total,
                "number_minorities": number_minorities,
                "number_ethnicities": number_ethnicities,
                "largest_groups": largest_groups,
            },
        ]
    return las


def get_df_ward_dict_dp(
    wards,
    column_names,
    epsilons,
    delta,
    sensitivity,
    clipping,
    rounding,
    random_state=None,
):
    wards_copy = deepcopy(wards)
    for code, ward in wards_copy.items():
        df = ward[0]

        # applying DP, one person can change count by +1/-1, therefore sensitivity should be 2 in this case
        for epsilon in epsilons:
            df = diff_priv.create_dp_column_laplace(
                df,
                column_names[0],
                f"{column_names[1]} {epsilon}",
                epsilon,
                delta,
                sensitivity,
                clipping,
                rounding,
                random_state,
            )

            # calculating DP percentages for checking population gain and loss
            total_dp = df[f"{column_names[1]} {epsilon}"].sum()
            df[f"{column_names[2]} {epsilon}"] = (
                df[f"{column_names[1]} {epsilon}"] / total_dp
            )

    return wards_copy


# this function takes the ward code as a seed
def get_df_ward_dict_dp_seed_ward(
    wards, table_name, column_names, epsilons, delta, sensitivity, clipping, rounding
):
    wards_copy = deepcopy(wards)
    table_num = "".join(filter(str.isdigit, table_name))
    for code, ward in wards_copy.items():
        df = ward[0]
        seed = int(code[1:3] + code[5:] + table_num)

        # applying DP, one person can change count by +1/-1, therefore sensitivity should be 2 in this case
        for epsilon in epsilons:
            df = diff_priv.create_dp_column_laplace(
                df,
                column_names[0],
                f"{column_names[1]} {epsilon}",
                epsilon,
                delta,
                sensitivity,
                clipping,
                rounding,
                seed,
            )

            # calculating DP percentages for checking population gain and loss
            total_dp = df[f"{column_names[1]} {epsilon}"].sum()
            df[f"{column_names[2]} {epsilon}"] = (
                df[f"{column_names[1]} {epsilon}"] / total_dp
            )

    return wards_copy


def get_df_ward_dict_dp_geometric(
    wards, column_names, epsilons, sensitivity, clipping, random_state=None
):
    wards_copy = deepcopy(wards)
    for code, ward in wards_copy.items():
        df = ward[0]

        # applying DP, one person can change count by +1/-1, therefore sensitivity should be 2 in this case
        for epsilon in epsilons:
            df = diff_priv.create_dp_column_geometric(
                df,
                column_names[0],
                f"{column_names[1]} {epsilon}",
                epsilon,
                sensitivity,
                clipping,
                random_state,
            )

            # calculating DP percentages for checking population gain and loss
            total_dp = df[f"{column_names[1]} {epsilon}"].sum()
            df[f"{column_names[2]} {epsilon}"] = (
                df[f"{column_names[1]} {epsilon}"] / total_dp
            )

    return wards_copy


def get_df_ward_dict_dp_geometric_seed_ward(
    wards, table_name, column_names, epsilons, sensitivity, clipping
):
    wards_copy = deepcopy(wards)
    table_num = "".join(filter(str.isdigit, table_name))
    for code, ward in wards_copy.items():
        df = ward[0]
        seed = int(code[1:3] + code[5:] + table_num)

        # applying DP, one person can change count by +1/-1, therefore sensitivity should be 2 in this case
        for epsilon in epsilons:
            df = diff_priv.create_dp_column_geometric(
                df,
                column_names[0],
                f"{column_names[1]} {epsilon}",
                epsilon,
                sensitivity,
                clipping,
                seed,
            )

            # calculating DP percentages for checking population gain and loss
            total_dp = df[f"{column_names[1]} {epsilon}"].sum()
            df[f"{column_names[2]} {epsilon}"] = (
                df[f"{column_names[1]} {epsilon}"] / total_dp
            )

    return wards_copy


def get_df_ward_dict_error(wards, df_area, sheet_cl):
    for code, ward in wards.items():
        la = df_area[df_area["GeographyCode"].str.startswith(code)]
        la_code = la["GeographyCodeLA"].values[0]
        cl_width_rel = get_la_cl(la_code, sheet_cl)

        df = ward[0]
        df = data_error.create_data_error_column(
            df, "PopulationNumbers", "PopulationNumbersDataError", cl_width_rel
        )
        total_error = df[f"PopulationNumbersDataError"].sum()
        df["data error %"] = df[f"PopulationNumbersDataError"] / total_error

    return wards


def get_la_cl(geography_code, sheet_cl):
    return sheet_cl[sheet_cl["area_code"].str.startswith(geography_code)][
        "relative_cl_width"
    ].values[0]


# dataframe which contains all metrics apart from kl-divergence
def create_metrics_df(wards, epsilons):
    measurement_dfs = []

    for code, ward in wards.items():

        df_measurements = pd.DataFrame(index=epsilons)
        df_measurements.index.name = "epsilon"

        df = ward[0]
        dict = ward[1]
        for epsilon in epsilons:

            total = df["PopulationNumbers"].sum()
            total_dp = df[f"PopulationNumbersDP {epsilon}"].sum()
            total_data_error = df["PopulationNumbersDataError"].sum()
            total_data_error_dp = df[f"PopulationNumbersDataErrorDP {epsilon}"].sum()

            rmse = root_mean_squared_error(
                df["PopulationNumbers"], df[f"PopulationNumbersDP {epsilon}"]
            )
            rmse_data_error = root_mean_squared_error(
                df["PopulationNumbers"], df["PopulationNumbersDataError"]
            )
            rmse_data_error_dp = root_mean_squared_error(
                df["PopulationNumbers"], df[f"PopulationNumbersDataErrorDP {epsilon}"]
            )

            decreased = df.loc[
                (df[f"significantly_decreased {epsilon}"]) & (df["total %"] > 0)
            ].shape[0]
            decreased_zero = df[f"significantly_decreased_zero {epsilon}"].sum()
            decreased_data_error = df.loc[
                (df["significantly_decreased_data_error"]) & (df["total %"] > 0)
            ].shape[0]
            decreased_data_error_zero = df[
                "significantly_decreased_data_error_zero"
            ].sum()
            decreased_data_error_dp = df.loc[
                (df[f"significantly_decreased_data_error_DP {epsilon}"])
                & (df["total %"] > 0)
            ].shape[0]
            decreased_data_error_dp_zero = df[
                f"significantly_decreased_data_error_DP_zero {epsilon}"
            ].sum()

            decreased_minority = df.loc[
                (df[f"significantly_decreased {epsilon}"])
                & (df["total %"] > 0)
                & (df["total %"] <= 0.1)
            ].shape[0]
            decreased_data_error_minority = df.loc[
                (df[f"significantly_decreased_data_error"])
                & (df["total %"] > 0)
                & (df["total %"] <= 0.1)
            ].shape[0]
            decreased_data_error_dp_minority = df.loc[
                (df[f"significantly_decreased_data_error_DP {epsilon}"])
                & (df["total %"] > 0)
                & (df["total %"] <= 0.1)
            ].shape[0]

            increased = df.loc[
                (df[f"significantly_increased {epsilon}"]) & (df["total %"] > 0)
            ].shape[0]
            increased_zero = df[f"significantly_increased_zero {epsilon}"].sum()
            increased_data_error = df.loc[
                (df["significantly_increased_data_error"]) & (df["total %"] > 0)
            ].shape[0]
            increased_data_error_zero = df[
                "significantly_increased_data_error_zero"
            ].sum()
            increased_data_error_dp = df.loc[
                (df[f"significantly_increased_data_error_DP {epsilon}"])
                & (df["total %"] > 0)
            ].shape[0]
            increased_data_error_dp_zero = df[
                f"significantly_increased_data_error_DP_zero {epsilon}"
            ].sum()

            increased_minority = df.loc[
                (df[f"significantly_increased {epsilon}"])
                & (df["total %"] > 0)
                & (df["total %"] <= 0.1)
            ].shape[0]
            increased_data_error_minority = df.loc[
                (df[f"significantly_increased_data_error"])
                & (df["total %"] > 0)
                & (df["total %"] <= 0.1)
            ].shape[0]
            increased_data_error_dp_minority = df.loc[
                (df[f"significantly_increased_data_error_DP {epsilon}"])
                & (df["total %"] > 0)
                & (df["total %"] <= 0.1)
            ].shape[0]

            df_measurements.at[epsilon, "area_name"] = dict["area_name"]
            df_measurements.at[epsilon, "number_minorities"] = dict["number_minorities"]
            df_measurements.at[epsilon, "number_ethnicities"] = dict[
                "number_ethnicities"
            ]
            df_measurements.at[epsilon, "diversity"] = classify_diversity(
                dict["number_minorities"]
            )
            df_measurements.at[epsilon, "area_code"] = code
            df_measurements.at[epsilon, "rmse_dp"] = rmse
            df_measurements.at[epsilon, "rmse_data_error"] = rmse_data_error
            df_measurements.at[epsilon, "rmse_data_error_dp"] = rmse_data_error_dp

            df_measurements.at[epsilon, "significantly_decreased"] = decreased
            df_measurements.at[epsilon, "significantly_decreased_data_error"] = (
                decreased_data_error
            )
            df_measurements.at[epsilon, "significantly_decreased_data_error_dp"] = (
                decreased_data_error_dp
            )
            df_measurements.at[epsilon, "significantly_increased"] = increased
            df_measurements.at[epsilon, "significantly_increased_data_error"] = (
                increased_data_error
            )
            df_measurements.at[epsilon, "significantly_increased_data_error_dp"] = (
                increased_data_error_dp
            )

            df_measurements.at[epsilon, "significantly_increased_zero"] = increased_zero
            df_measurements.at[epsilon, "significantly_increased_data_error_zero"] = (
                increased_data_error_zero
            )
            df_measurements.at[
                epsilon, "significantly_increased_data_error_dp_zero"
            ] = increased_data_error_dp_zero

            df_measurements.at[epsilon, "significantly_decreased_zero"] = decreased_zero
            df_measurements.at[epsilon, "significantly_decreased_data_error_zero"] = (
                decreased_data_error_zero
            )
            df_measurements.at[
                epsilon, "significantly_decreased_data_error_dp_zero"
            ] = decreased_data_error_dp_zero

            df_measurements.at[epsilon, "significantly_decreased_minority"] = (
                decreased_minority
            )
            df_measurements.at[
                epsilon, "significantly_decreased_data_error_minority"
            ] = decreased_data_error_minority
            df_measurements.at[
                epsilon, "significantly_decreased_data_error_dp_minority"
            ] = decreased_data_error_dp_minority
            df_measurements.at[epsilon, "significantly_increased_minority"] = (
                increased_minority
            )
            df_measurements.at[
                epsilon, "significantly_increased_data_error_minority"
            ] = increased_data_error_minority
            df_measurements.at[
                epsilon, "significantly_increased_data_error_dp_minority"
            ] = increased_data_error_dp_minority

            df_measurements.at[epsilon, "total"] = total
            df_measurements.at[epsilon, "total_dp"] = total_dp
            df_measurements.at[epsilon, "total_data_error"] = total_data_error
            df_measurements.at[epsilon, "total_data_error_dp"] = total_data_error_dp

        measurement_dfs.append(df_measurements)

    measurement_df = pd.concat(measurement_dfs)
    measurement_df.reset_index()

    return measurement_df


# measures rmse for several values of epsilon
def measure_rmse(wards, epsilons):
    measurement_dfs = []

    for code, ward in wards.items():

        df_measurements = pd.DataFrame(index=epsilons)
        df_measurements.index.name = "epsilon"

        df = ward[0]
        dict = ward[1]

        for epsilon in epsilons:

            rmse = root_mean_squared_error(
                df["PopulationNumbers"], df[f"PopulationNumbersDP {epsilon}"]
            )
            df_measurements.at[epsilon, "area_name"] = dict["area_name"]
            df_measurements.at[epsilon, "number_minorities"] = dict["number_minorities"]
            df_measurements.at[epsilon, "diversity"] = classify_diversity(
                dict["number_minorities"]
            )
            df_measurements.at[epsilon, "area_code"] = code
            df_measurements.at[epsilon, "rmse"] = rmse

        measurement_dfs.append(df_measurements)

    measurement_df = pd.concat(measurement_dfs)
    measurement_df.reset_index()

    return measurement_df


# measures mse for data_error
def measure_rmse_data_error(wards, epsilons):
    measurement_dfs = []

    for code, ward in wards.items():
        df_measurements = pd.DataFrame(index=epsilons)
        df_measurements.index.name = "epsilon"

        df = ward[0]
        dict = ward[1]

        for epsilon in epsilons:

            rmse = root_mean_squared_error(
                df["PopulationNumbers"], df["PopulationNumbersDataError"]
            )
            df_measurements.at[epsilon, "area_name"] = dict["area_name"]
            df_measurements.at[epsilon, "number_minorities"] = dict["number_minorities"]
            df_measurements.at[epsilon, "diversity"] = classify_diversity(
                dict["number_minorities"]
            )
            df_measurements.at[epsilon, "area_code"] = code
            df_measurements.at[epsilon, "rmse"] = rmse

        measurement_dfs.append(df_measurements)

    measurement_df = pd.concat(measurement_dfs)
    measurement_df.reset_index()

    return measurement_df


# the kl divergence can only be calculated if clipping is applied, otherwise events might have a negative probability in the distribution
def measure_kl_divergence(wards, ward_codes, epsilons, mus):
    measurement_dfs = []

    for code, ward in wards.items():

        index = pd.MultiIndex.from_product([epsilons, mus], names=["epsilon", "mu"])
        df_measurements = pd.DataFrame(index=index)

        df = ward[0]
        dict = ward[1]
        for epsilon in epsilons:
            for mu in mus:
                kl_divergence = mu_smoothed_kl_divergence(
                    df["%"].values.tolist(), df[f"dp % {epsilon}"].values.tolist(), mu
                )
                kl_divergence_data_error = mu_smoothed_kl_divergence(
                    df["%"].values.tolist(), df["data error %"].values.tolist(), mu
                )
                kl_divergence_data_error_dp = mu_smoothed_kl_divergence(
                    df["%"].values.tolist(),
                    df[f"dp data error % {epsilon}"].values.tolist(),
                    mu,
                )

                df_measurements.at[(epsilon, mu), "area_name"] = dict["area_name"]
                df_measurements.at[(epsilon, mu), "number_minorities"] = dict[
                    "number_minorities"
                ]
                df_measurements.at[(epsilon, mu), "diversity"] = classify_diversity(
                    dict["number_minorities"]
                )
                df_measurements.at[(epsilon, mu), "area_code"] = code
                df_measurements.at[(epsilon, mu), "kl_divergence"] = kl_divergence
                df_measurements.at[(epsilon, mu), "kl_divergence_data_error"] = (
                    kl_divergence_data_error
                )
                df_measurements.at[(epsilon, mu), "kl_divergence_data_error_dp"] = (
                    kl_divergence_data_error_dp
                )

        measurement_dfs.append(df_measurements)

    measurement_df = pd.concat(measurement_dfs)
    measurement_df.reset_index()

    return measurement_df


# the kl divergence can only be calculated if clipping is applied, otherwise events might have a negative probability in the distribution
def measure_kl_divergence_experiments(experiments, ward_codes, epsilons, mus):
    kl_experiments = np.empty(len(experiments), dtype=object)
    for i in range(len(experiments)):
        measurement_dfs = []
        for code in ward_codes:
            ward = experiments[i]["wards_dp"][code]

            index = pd.MultiIndex.from_product([epsilons, mus], names=["epsilon", "mu"])
            df_measurements = pd.DataFrame(index=index)

            df = ward[0]
            dict = ward[1]
            for epsilon in epsilons:
                for mu in mus:
                    kl_divergence = mu_smoothed_kl_divergence(
                        df["%"].values.tolist(),
                        df[f"dp % {epsilon}"].values.tolist(),
                        mu,
                    )
                    kl_divergence_data_error = mu_smoothed_kl_divergence(
                        df["%"].values.tolist(), df["data error %"].values.tolist(), mu
                    )
                    kl_divergence_data_error_dp = mu_smoothed_kl_divergence(
                        df["%"].values.tolist(),
                        df[f"dp data error % {epsilon}"].values.tolist(),
                        mu,
                    )

                    df_measurements.at[(epsilon, mu), "area_name"] = dict["area_name"]
                    df_measurements.at[(epsilon, mu), "number_minorities"] = dict[
                        "number_minorities"
                    ]
                    df_measurements.at[(epsilon, mu), "diversity"] = classify_diversity(
                        dict["number_minorities"]
                    )
                    df_measurements.at[(epsilon, mu), "area_code"] = code
                    df_measurements.at[(epsilon, mu), "kl_divergence"] = kl_divergence
                    df_measurements.at[(epsilon, mu), "kl_divergence_data_error"] = (
                        kl_divergence_data_error
                    )
                    df_measurements.at[(epsilon, mu), "kl_divergence_data_error_dp"] = (
                        kl_divergence_data_error_dp
                    )

            measurement_dfs.append(df_measurements)

        measurement_df = pd.concat(measurement_dfs)
        measurement_df.reset_index()
        kl_experiments[i] = measurement_df

    return kl_experiments


def create_overview(dict):
    dicts = []
    for code in dict:
        ward = dict[code]
        dicts.append(ward[1] | {"area_code": code})
    dicts
    df_overview = pd.DataFrame(dicts)
    return df_overview


# calculates whether a population group got significantly decreased when applying noise. A group counts as significantly decreased if it is less than 75% of it's original size
def significantly_decreased(p, q):
    x = 0.001
    if p == 0:
        # if q is 0, I am adding a small constant x to p and q to calculate the increase
        return (q + x) < (0.75 * (p + x))
    else:
        return q < 0.75 * p


# calculates whether a population group got significantly increased when applying noise. A group counts as significantly increased if it is larger than 125% of it's original size
def significantly_increased(p, q):
    # adding x value to p and q in case one of the values is 0
    x = 0.001
    if p == 0:
        return (q + x) > (1.25 * (p + x))
    else:
        return q > 1.25 * p


def get_significantly_decreased_increased(wards, epsilons):
    for code, ward in wards.items():
        df = ward[0]

        for epsilon in epsilons:
            # adding a significantly decreased column to dataframe
            df[f"significantly_decreased {epsilon}"] = df.apply(
                lambda x: significantly_decreased(
                    x["PopulationNumbers"], x[f"PopulationNumbersDP {epsilon}"]
                ),
                axis=1,
            )
            df[f"significantly_decreased_zero {epsilon}"] = np.where(
                (df[f"significantly_decreased {epsilon}"] == True)
                & (df["PopulationNumbers"] == 0),
                True,
                False,
            )
            df[f"significantly_increased {epsilon}"] = df.apply(
                lambda x: significantly_increased(
                    x["PopulationNumbers"], x[f"PopulationNumbersDP {epsilon}"]
                ),
                axis=1,
            )
            df[f"significantly_increased_zero {epsilon}"] = np.where(
                (df[f"significantly_increased {epsilon}"] == True)
                & (df["PopulationNumbers"] == 0),
                True,
                False,
            )

    return wards


def get_significantly_decreased_increased_data_error(wards):
    for code, ward in wards.items():
        df = ward[0]
        # adding a significantly decreased column to dataframe
        df["significantly_decreased_data_error"] = df.apply(
            lambda x: significantly_decreased(
                x["PopulationNumbers"], x[f"PopulationNumbersDataError"]
            ),
            axis=1,
        )
        df[f"significantly_decreased_data_error_zero"] = np.where(
            (df[f"significantly_decreased_data_error"] == True)
            & (df["PopulationNumbers"] == 0),
            True,
            False,
        )
        df["significantly_increased_data_error"] = df.apply(
            lambda x: significantly_increased(
                x["PopulationNumbers"], x[f"PopulationNumbersDataError"]
            ),
            axis=1,
        )
        df[f"significantly_increased_data_error_zero"] = np.where(
            (df[f"significantly_increased_data_error"] == True)
            & (df["PopulationNumbers"] == 0),
            True,
            False,
        )

    return wards


def get_significantly_decreased_increased_data_error_DP(wards, epsilons):
    for code, ward in wards.items():
        df = ward[0]
        # adding a significantly decreased column to dataframe
        for epsilon in epsilons:
            df[f"significantly_decreased_data_error_DP {epsilon}"] = df.apply(
                lambda x: significantly_decreased(
                    x["PopulationNumbers"], x[f"PopulationNumbersDataErrorDP {epsilon}"]
                ),
                axis=1,
            )
            df[f"significantly_decreased_data_error_DP_zero {epsilon}"] = np.where(
                (df[f"significantly_decreased_data_error_DP {epsilon}"] == True)
                & (df["PopulationNumbers"] == 0),
                True,
                False,
            )
            df[f"significantly_increased_data_error_DP {epsilon}"] = df.apply(
                lambda x: significantly_increased(
                    x["PopulationNumbers"], x[f"PopulationNumbersDataErrorDP {epsilon}"]
                ),
                axis=1,
            )
            df[f"significantly_increased_data_error_DP_zero {epsilon}"] = np.where(
                (df[f"significantly_increased_data_error_DP {epsilon}"] == True)
                & (df["PopulationNumbers"] == 0),
                True,
                False,
            )

    return wards


def get_average_decreased(n, experiments):
    results = np.empty(n, dtype=object)
    for i in range(n):
        results[i] = experiments[i]["decreased_df"][
            "significantly_decreased"
        ].to_numpy()
    return np.average(results, axis=0)


# counts number of significantly decreased groups for several values of epsilon
def measure_average_significantly_decreased(average, wards, epsilons):
    measurement_dfs = []

    for code, ward in wards.items():

        df_measurements = pd.DataFrame(index=epsilons)
        df_measurements.index.name = "epsilon"

        df = ward[0]
        dict = ward[1]
        for epsilon in epsilons:

            df_measurements.at[epsilon, "area_name"] = dict["area_name"]
            df_measurements.at[epsilon, "number_minorities"] = dict["number_minorities"]
            df_measurements.at[epsilon, "area_code"] = code

        measurement_dfs.append(df_measurements)

    measurement_df = pd.concat(measurement_dfs)
    measurement_df.reset_index()
    measurement_df["average_significantly_decreased"] = average

    return measurement_df


# filters data with a filter dict, returns reduced lookup table, the names of the reduced datasets and the reduced csv
def get_reduced_data(lookup, filter_dict, csv):
    reduced_lookup = lookup.loc[
        lookup[list(filter_dict)].isin(filter_dict).all(axis=1), :
    ]
    datasets_reduced = reduced_lookup.Dataset.values.tolist()
    reduced_csv = csv[["GeographyCode"] + datasets_reduced]
    return reduced_lookup, datasets_reduced, reduced_csv


def set_up_measurements_wards(
    wards,
    df_area,
    sheet_cl,
    mechanism,
    epsilons,
    delta,
    sensitivity,
    clipping,
    rounding,
    random_state=None,
):
    column_names_dp = ["PopulationNumbers", "PopulationNumbersDP", "dp %"]
    column_names_dp_data = [
        "PopulationNumbersDataError",
        "PopulationNumbersDataErrorDP",
        "dp data error %",
    ]
    wards_dp = get_df_ward_dict_error(wards, df_area, sheet_cl)
    if mechanism == "laplace":
        wards_dp = get_df_ward_dict_dp(
            wards_dp,
            column_names_dp,
            epsilons,
            delta,
            sensitivity,
            clipping,
            rounding,
            random_state,
        )
        wards_dp = get_df_ward_dict_dp(
            wards_dp,
            column_names_dp_data,
            epsilons,
            delta,
            sensitivity,
            clipping,
            rounding,
            random_state,
        )

    elif mechanism == "geometric":
        wards_dp = get_df_ward_dict_dp_geometric(
            wards_dp, column_names_dp, epsilons, sensitivity, clipping, random_state
        )
        # now DP is applied to data error column
        wards_dp = get_df_ward_dict_dp_geometric(
            wards_dp,
            column_names_dp_data,
            epsilons,
            sensitivity,
            clipping,
            random_state,
        )
    else:
        print(f"{mechanism} is not a valid argument")
        return None

    wards_dp = get_significantly_decreased_increased(wards_dp, epsilons)
    wards_dp = get_significantly_decreased_increased_data_error(wards_dp)
    wards_dp = get_significantly_decreased_increased_data_error_DP(wards_dp, epsilons)
    metrics_df = create_metrics_df(wards_dp, epsilons)

    return (wards_dp, metrics_df)


# the experiment is repeated n times
def set_up_measurements_wards_repeat(
    n,
    wards,
    df_area,
    sheet_cl,
    mechanism,
    epsilons,
    delta,
    sensitivity,
    clipping,
    rounding,
):
    experiments = np.empty(n, dtype=object)

    # here random state is set to i so that the same amount of DP noise is applied on the ground truth and the data error data set

    for i in range(n):
        column_names_dp = ["PopulationNumbers", "PopulationNumbersDP", "dp %"]
        column_names_dp_data = [
            "PopulationNumbersDataError",
            "PopulationNumbersDataErrorDP",
            "dp data error %",
        ]
        wards_dp = get_df_ward_dict_error(wards, df_area, sheet_cl)

        if mechanism == "laplace":
            wards_dp = get_df_ward_dict_dp(
                wards_dp,
                column_names_dp,
                epsilons,
                delta,
                sensitivity,
                clipping,
                rounding,
                random_state=i,
            )
            wards_dp = get_df_ward_dict_dp(
                wards_dp,
                column_names_dp_data,
                epsilons,
                delta,
                sensitivity,
                clipping,
                rounding,
                random_state=i,
            )
        elif mechanism == "geometric":
            wards_dp = get_df_ward_dict_dp_geometric(
                wards, column_names_dp, epsilons, sensitivity, clipping, random_state=i
            )
            wards_dp = get_df_ward_dict_dp_geometric(
                wards_dp,
                column_names_dp_data,
                epsilons,
                sensitivity,
                clipping,
                random_state=i,
            )
        else:
            print(f"{mechanism} is not a valid argument")
            return None

        wards_dp = get_significantly_decreased_increased(wards_dp, epsilons)
        wards_dp = get_significantly_decreased_increased_data_error(wards_dp)
        wards_dp = get_significantly_decreased_increased_data_error_DP(
            wards_dp, epsilons
        )
        metrics_df = create_metrics_df(wards_dp, epsilons)

        experiments[i] = {"wards_dp": wards_dp, "metrics_df": metrics_df}

    return experiments


# the experiment is repeated n times, a random number for each experiment i is used for the seeding of the random state
def set_up_measurements_wards_repeat_random_seed(
    n,
    wards,
    df_area,
    sheet_cl,
    mechanism,
    epsilons,
    delta,
    sensitivity,
    clipping,
    rounding,
):
    experiments = np.empty(n, dtype=object)

    for i in range(n):
        # creating a positive 32 bit seed for each experiment i to pass as a random_state
        seed = random.randint(0, 2147483647)

        column_names_dp = ["PopulationNumbers", "PopulationNumbersDP", "dp %"]
        column_names_dp_data = [
            "PopulationNumbersDataError",
            "PopulationNumbersDataErrorDP",
            "dp data error %",
        ]
        wards_dp = get_df_ward_dict_error(wards, df_area, sheet_cl)

        if mechanism == "laplace":
            wards_dp = get_df_ward_dict_dp(
                wards_dp,
                column_names_dp,
                epsilons,
                delta,
                sensitivity,
                clipping,
                rounding,
                random_state=seed,
            )
            wards_dp = get_df_ward_dict_dp(
                wards_dp,
                column_names_dp_data,
                epsilons,
                delta,
                sensitivity,
                clipping,
                rounding,
                random_state=seed,
            )
        elif mechanism == "geometric":
            wards_dp = get_df_ward_dict_dp_geometric(
                wards,
                column_names_dp,
                epsilons,
                sensitivity,
                clipping,
                random_state=seed,
            )
            wards_dp = get_df_ward_dict_dp_geometric(
                wards_dp,
                column_names_dp_data,
                epsilons,
                sensitivity,
                clipping,
                random_state=seed,
            )
        else:
            print(f"{mechanism} is not a valid argument")
            return None

        wards_dp = get_significantly_decreased_increased(wards_dp, epsilons)
        wards_dp = get_significantly_decreased_increased_data_error(wards_dp)
        wards_dp = get_significantly_decreased_increased_data_error_DP(
            wards_dp, epsilons
        )
        metrics_df = create_metrics_df(wards_dp, epsilons)

        experiments[i] = {"wards_dp": wards_dp, "metrics_df": metrics_df}

    return experiments


# the experiment is repeated n times, here no random state is used
def set_up_measurements_wards_repeat_no_rand(
    n,
    wards,
    df_area,
    sheet_cl,
    mechanism,
    epsilons,
    delta,
    sensitivity,
    clipping,
    rounding,
):
    experiments = np.empty(n, dtype=object)

    for i in range(n):
        column_names_dp = ["PopulationNumbers", "PopulationNumbersDP", "dp %"]
        column_names_dp_data = [
            "PopulationNumbersDataError",
            "PopulationNumbersDataErrorDP",
            "dp data error %",
        ]
        wards_dp = get_df_ward_dict_error(wards, df_area, sheet_cl)

        if mechanism == "laplace":
            wards_dp = get_df_ward_dict_dp(
                wards_dp,
                column_names_dp,
                epsilons,
                delta,
                sensitivity,
                clipping,
                rounding,
                random_state=None,
            )
            wards_dp = get_df_ward_dict_dp(
                wards_dp,
                column_names_dp_data,
                epsilons,
                delta,
                sensitivity,
                clipping,
                rounding,
                random_state=None,
            )
        elif mechanism == "geometric":
            wards_dp = get_df_ward_dict_dp_geometric(
                wards,
                column_names_dp,
                epsilons,
                sensitivity,
                clipping,
                random_state=None,
            )
            wards_dp = get_df_ward_dict_dp_geometric(
                wards_dp,
                column_names_dp_data,
                epsilons,
                sensitivity,
                clipping,
                random_state=None,
            )
        else:
            print(f"{mechanism} is not a valid argument")
            return None

        wards_dp = get_significantly_decreased_increased(wards_dp, epsilons)
        wards_dp = get_significantly_decreased_increased_data_error(wards_dp)
        wards_dp = get_significantly_decreased_increased_data_error_DP(
            wards_dp, epsilons
        )
        metrics_df = create_metrics_df(wards_dp, epsilons)

        experiments[i] = {"wards_dp": wards_dp, "metrics_df": metrics_df}

    return experiments


def calculate_indicators_tenure(wards, indicator_dict):
    wards_copy = deepcopy(wards)

    for code in wards_copy:
        ward_df = wards_copy[code][0]
        total_not_owned = ward_df.query('Tenure != "Owned or shared ownership: Total"')[
            "PopulationNumbers"
        ].sum()
        total_tenure = ward_df["PopulationNumbers"].sum()
        percentage_tenure = (total_not_owned / total_tenure) * 100
        indicator_dict[code] |= {"percentage_tenure": percentage_tenure}

    return indicator_dict


def calculate_indicators_tenure_dp(wards, indicator_dict, epsilons):
    wards_copy = deepcopy(wards)

    for code in wards_copy:
        ward_df = wards_copy[code][0]
        for epsilon in epsilons:
            total_not_owned = ward_df.query(
                'Tenure != "Owned or shared ownership: Total"'
            )[f"PopulationNumbersDP {epsilon}"].sum()
            total_tenure = ward_df[f"PopulationNumbersDP {epsilon}"].sum()
            percentage_tenure = (total_not_owned / total_tenure) * 100
            indicator_dict[code] |= {f"percentage_tenure_{epsilon}": percentage_tenure}

    return indicator_dict


def calculate_indicators_occ(wards, indicator_dict):
    wards_copy = deepcopy(wards)
    for code in wards_copy:
        ward_df = wards_copy[code][0]
        total_overcrowded = ward_df.query(
            'Occupancy == "Occupancy rating (bedrooms) of -1 or less"'
        )["PopulationNumbers"].sum()
        total_occ = ward_df["PopulationNumbers"].sum()
        percentage_overcrowding = (total_overcrowded / total_occ) * 100
        # applying log transformation here to normalise skewed data
        log_percentage_overcrowding = np.log(percentage_overcrowding + 1)
        indicator_dict[code] |= {
            "log_percentage_overcrowding": log_percentage_overcrowding
        }

    return indicator_dict


def calculate_indicators_occ_dp(wards, indicator_dict, epsilons):
    wards_copy = deepcopy(wards)
    for code in wards_copy:
        ward_df = wards_copy[code][0]
        for epsilon in epsilons:
            total_overcrowded = ward_df.query(
                'Occupancy == "Occupancy rating (bedrooms) of -1 or less"'
            )[f"PopulationNumbersDP {epsilon}"].sum()

            total_occ = ward_df[f"PopulationNumbersDP {epsilon}"].sum()

            percentage_overcrowding = (total_overcrowded / total_occ) * 100
            # applying log transformation here to normalise skewed data
            log_percentage_overcrowding = np.log(percentage_overcrowding + 1)
            indicator_dict[code] |= {
                f"log_percentage_overcrowding_{epsilon}": log_percentage_overcrowding
            }

    return indicator_dict


def calculate_indicators_car(wards, indicator_dict):
    wards_copy = deepcopy(wards)
    for code in wards_copy:

        ward_df = wards_copy[code][0]
        total_no_car = ward_df.query('Car == "No cars or vans in household"')[
            "PopulationNumbers"
        ].sum()

        total_car = ward_df["PopulationNumbers"].sum()

        percentage_no_car = (total_no_car / total_car) * 100
        indicator_dict[code] |= {"percentage_no_car": percentage_no_car}

    return indicator_dict


def calculate_indicators_car_dp(wards, indicator_dict, epsilons):
    wards_copy = deepcopy(wards)
    for code in wards_copy:

        ward_df = wards_copy[code][0]
        for epsilon in epsilons:
            total_no_car = ward_df.query('Car == "No cars or vans in household"')[
                f"PopulationNumbersDP {epsilon}"
            ].sum()

            total_car = ward_df[f"PopulationNumbersDP {epsilon}"].sum()

            percentage_no_car = (total_no_car / total_car) * 100
            indicator_dict[code] |= {f"percentage_no_car_{epsilon}": percentage_no_car}

    return indicator_dict


def calculate_indicators_economic(wards, indicator_dict):
    wards_copy = deepcopy(wards)
    for code in wards_copy:

        ward_df = wards_copy[code][0]
        total_unemployment = ward_df.query(
            'EconomicActivity == "Economically active: Unemployed: Unemployed (excluding full time students)"'
        )["PopulationNumbers"].sum()

        total_economically_active = ward_df.query(
            'EconomicActivity.str.startswith("Economically active")'
        )["PopulationNumbers"].sum()

        percentage_unemployment = (total_unemployment / total_economically_active) * 100

        log_percentage_unemployment = np.log(percentage_unemployment + 1)
        indicator_dict[code] |= {
            "log_percentage_unemployment": log_percentage_unemployment
        }

    return indicator_dict


def calculate_indicators_economic_dp(wards, indicator_dict, epsilons):
    wards_copy = deepcopy(wards)
    for code in wards_copy:

        ward_df = wards_copy[code][0]
        for epsilon in epsilons:
            total_unemployment = ward_df.query(
                'EconomicActivity == "Economically active: Unemployed: Unemployed (excluding full time students)"'
            )[f"PopulationNumbersDP {epsilon}"].sum()

            total_economically_active = ward_df.query(
                'EconomicActivity.str.startswith("Economically active")'
            )[f"PopulationNumbersDP {epsilon}"].sum()

            percentage_unemployment = (
                total_unemployment / total_economically_active
            ) * 100

            log_percentage_unemployment = np.log(percentage_unemployment + 1)
            indicator_dict[code] |= {
                f"log_percentage_unemployment_{epsilon}": log_percentage_unemployment
            }

    return indicator_dict


def get_all_indicators(indicators):
    all_tenure_indicators = []
    all_overcrowding_indicators = []
    all_car_indicators = []
    all_unemployment_indicators = []

    for key, value in indicators.items():
        all_tenure_indicators.append(value["percentage_tenure"])
        all_overcrowding_indicators.append(value["log_percentage_overcrowding"])
        all_car_indicators.append(value["percentage_no_car"])
        all_unemployment_indicators.append(value["log_percentage_unemployment"])

    return (
        all_tenure_indicators,
        all_overcrowding_indicators,
        all_car_indicators,
        all_unemployment_indicators,
    )


def get_all_indicators_dp(indicators, epsilons):
    all_indicator_dict = dict([(key, {}) for key in epsilons])
    for epsilon in epsilons:
        all_tenure_indicators = []
        all_overcrowding_indicators = []
        all_car_indicators = []
        all_unemployment_indicators = []

        for key, value in indicators.items():
            all_tenure_indicators.append(value[f"percentage_tenure_{epsilon}"])
            all_overcrowding_indicators.append(
                value[f"log_percentage_overcrowding_{epsilon}"]
            )
            all_car_indicators.append(value[f"percentage_no_car_{epsilon}"])
            all_unemployment_indicators.append(
                value[f"log_percentage_unemployment_{epsilon}"]
            )

        all_indicator_dict[epsilon] = {
            "all_tenure_indicators": all_tenure_indicators,
            "all_overcrowding_indicators": all_overcrowding_indicators,
            "all_car_indicators": all_car_indicators,
            "all_unemployment_indicators": all_unemployment_indicators,
        }

    return all_indicator_dict


# any areas with a score above zero are above the mean and therefore deprived, areas with scores below zero are less deprived
def calculate_deprivation_indices(indicators):
    (
        all_tenure_indicators,
        all_overcrowding_indicators,
        all_car_indicators,
        all_unemployment_indicators,
    ) = get_all_indicators(indicators)

    deprivation_indices = {}

    std_tenure = np.std(all_tenure_indicators)
    mean_tenure = np.mean(all_tenure_indicators)

    std_overcrowding = np.std(all_overcrowding_indicators)
    mean_overcrowding = np.mean(all_overcrowding_indicators)

    std_car = np.std(all_car_indicators)
    mean_car = np.mean(all_car_indicators)

    std_unemployment = np.std(all_unemployment_indicators)
    mean_unemployment = np.mean(all_unemployment_indicators)

    for key, value in indicators.items():
        indicator_tenure = value["percentage_tenure"]
        z_score_tenure = (indicator_tenure - mean_tenure) / std_tenure

        indicator_overcrowding = value["log_percentage_overcrowding"]
        z_score_overcrowding = (
            indicator_overcrowding - mean_overcrowding
        ) / std_overcrowding

        indicator_car = value["percentage_no_car"]
        z_score_car = (indicator_car - mean_car) / std_car

        indicator_unemployment = value["log_percentage_unemployment"]
        z_score_unemployment = (
            indicator_unemployment - mean_unemployment
        ) / std_unemployment

        deprivation_score = (
            z_score_tenure + z_score_overcrowding + z_score_car + z_score_unemployment
        )

        deprivation_indices[key] = deprivation_score

    return deprivation_indices


# any areas with a score above zero are above the mean and therefore deprived, areas with scores below zero are less deprived
def calculate_deprivation_indices_dp(indicators, epsilons):
    all_indicator_dict = get_all_indicators_dp(indicators, epsilons)

    all_deprivation_indices = {}

    for epsilon in epsilons:
        deprivation_indices = {}
        indicator_dict = all_indicator_dict[epsilon]

        std_tenure = np.std(indicator_dict["all_tenure_indicators"])
        mean_tenure = np.mean(indicator_dict["all_tenure_indicators"])

        std_overcrowding = np.std(indicator_dict["all_overcrowding_indicators"])
        mean_overcrowding = np.mean(indicator_dict["all_overcrowding_indicators"])

        std_car = np.std(indicator_dict["all_car_indicators"])
        mean_car = np.mean(indicator_dict["all_car_indicators"])

        std_unemployment = np.std(indicator_dict["all_unemployment_indicators"])
        mean_unemployment = np.mean(indicator_dict["all_unemployment_indicators"])

        for key, value in indicators.items():
            indicator_tenure = value[f"percentage_tenure_{epsilon}"]
            z_score_tenure = (indicator_tenure - mean_tenure) / std_tenure

            indicator_overcrowding = value[f"log_percentage_overcrowding_{epsilon}"]
            z_score_overcrowding = (
                indicator_overcrowding - mean_overcrowding
            ) / std_overcrowding

            indicator_car = value[f"percentage_no_car_{epsilon}"]
            z_score_car = (indicator_car - mean_car) / std_car

            indicator_unemployment = value[f"log_percentage_unemployment_{epsilon}"]
            z_score_unemployment = (
                indicator_unemployment - mean_unemployment
            ) / std_unemployment

            deprivation_score = (
                z_score_tenure
                + z_score_overcrowding
                + z_score_car
                + z_score_unemployment
            )

            deprivation_indices[key] = deprivation_score

        all_deprivation_indices[epsilon] = deprivation_indices

    return all_deprivation_indices


def set_up_deprivation_df(deprivation_indices, wards, path, name):
    measurement_dfs = []

    for code, ward in wards.items():

        df_measurements = {}

        dict = ward[1]

        df_measurements["area_name"] = dict["area_name"]
        df_measurements["number_minorities"] = dict["number_minorities"]
        df_measurements["number_ethnicities"] = dict["number_ethnicities"]
        df_measurements["diversity"] = classify_diversity(dict["number_ethnicities"])
        df_measurements["area_code"] = code
        df_measurements["deprivation_score"] = deprivation_indices[code]
        measurement_dfs.append(df_measurements)

    measurement_df = pd.DataFrame(measurement_dfs)
    measurement_df["quantile"] = pd.qcut(
        measurement_df["deprivation_score"], 5, labels=False
    )
    measurement_df.sort_values(by=["quantile"], inplace=True)
    measurement_df.reset_index()

    measurement_df.to_csv(os.path.join(path, name), index=False)

    return None


def set_up_deprivation_df_dp(deprivation_indices, wards, epsilons, path, dp_mechanism):
    deprivation_dict = {}
    for epsilon in epsilons:
        measurement_dfs = []

        for code, ward in wards.items():

            df_measurements = {}

            dict = ward[1]

            df_measurements["area_name"] = dict["area_name"]
            df_measurements["number_minorities"] = dict["number_minorities"]
            df_measurements["number_ethnicities"] = dict["number_ethnicities"]
            df_measurements["diversity"] = classify_diversity(
                dict["number_ethnicities"]
            )
            df_measurements["area_code"] = code
            df_measurements["deprivation_score"] = deprivation_indices[epsilon][code]
            measurement_dfs.append(df_measurements)

        measurement_df = pd.DataFrame(measurement_dfs)
        measurement_df["quantile"] = pd.qcut(
            measurement_df["deprivation_score"], 5, labels=False
        )
        measurement_df.sort_values(by=["quantile"], inplace=True)
        measurement_df.reset_index()

        measurement_df.to_csv(
            os.path.join(path, f"{dp_mechanism}_{epsilon}_dp.csv"), index=False
        )

    return None


def read_dp_csvs(epsilons, path, table_name):
    csv_dict = {}
    for epsilon in epsilons:
        df = pd.read_csv(os.path.join(path, f"{table_name}_{epsilon}_dp.csv"))
        csv_dict[epsilon] = df
    return csv_dict


def get_csv_dp_dict(wards, epsilons, path, table_name):
    wards_dp = deepcopy(wards)

    dp_csvs = read_dp_csvs(epsilons, path, table_name)
    for code in wards_dp:
        dp_df = wards_dp[code][0]
        for epsilon in epsilons:
            col = dp_csvs[epsilon][
                dp_csvs[epsilon]["GeographyCode"].isin([code])
            ].values.tolist()[0][4:]
            dp_df[f"PopulationNumbersDP {epsilon}"] = col
    return wards_dp


def get_csv_dp_dict_la(las, epsilons, path, table_name):
    las_dp = deepcopy(las)

    dp_csvs = read_dp_csvs(epsilons, path, table_name)
    for code in las_dp:
        dp_df = las_dp[code][0]
        for epsilon in epsilons:
            col = dp_csvs[epsilon][
                dp_csvs[epsilon]["GeographyCode"].isin([code])
            ].values.tolist()[0][2:]
            dp_df[f"PopulationNumbersDP {epsilon}"] = col
    return las_dp


def get_csv_dict_data_error(wards, path, table_name):
    wards_copy = deepcopy(wards)
    csv = pd.read_csv(os.path.join(path, f"{table_name}_data_error.csv"))
    for code in wards_copy:
        df = wards_copy[code][0]

        col = csv[csv["GeographyCode"].isin([code])].values.tolist()[0][4:]
        df[f"PopulationNumbersDataError"] = col
    return wards_copy


def get_csv_dict_data_error_la(las, path, table_name):
    las_copy = deepcopy(las)
    csv = pd.read_csv(os.path.join(path, f"{table_name}_data_error.csv"))
    for code in las_copy:
        df = las_copy[code][0]
        col = csv[csv["GeographyCode"].isin([code])].values.tolist()[0][2:]
        df[f"PopulationNumbersDataError"] = col
    return las_copy


def merge_dfs_dp(df_dp, df_base):
    dfs = []
    df_base.sort_values(by=["area_code"], inplace=True)
    df_base.reset_index(inplace=True, drop=True)

    for code, df in df_dp.items():
        df.sort_values(by=["area_code"], inplace=True)
        df.reset_index(inplace=True, drop=True)
        df["epsilon"] = code
        df["change_in_score"] = abs(
            df_base["deprivation_score"] - df["deprivation_score"]
        )

        df["change_in_quantile"] = df["quantile"] - df_base["quantile"]
        dfs.append(df)

    df_copy = df_base.copy(deep=True)
    df_copy["epsilon"] = 0
    df_copy["change_in_score"] = 0
    df_copy["change_in_quantile"] = 0
    dfs.append(df_copy)

    measurement_df = pd.concat(dfs)
    measurement_df.set_index("epsilon", inplace=True)
    return measurement_df


def calculate_changes(df, df_base):
    df_base.sort_values(by=["area_code"], inplace=True)
    df_base.reset_index(inplace=True, drop=True)

    df.sort_values(by=["area_code"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df["change_in_score"] = abs(df_base["deprivation_score"] - df["deprivation_score"])

    df["change_in_quantile"] = df["quantile"] - df_base["quantile"]

    return df


def make_experiment_df(experiments, ward_names, column_names, epsilons):
    data = []
    for name in ward_names:

        val_list = []
        for i in range(len(experiments)):
            metrics = (
                experiments[i]["metrics_df"].query("area_name == @name").copy(deep=True)
            )
            for epsilon in epsilons:
                for column_name in column_names:
                    val_list.append(
                        (
                            name,
                            epsilon,
                            column_name,
                            metrics.loc[epsilon][column_name].tolist(),
                        )
                    )
        data.extend(val_list)

    df = pd.DataFrame(data=data, columns=["area_name", "epsilon", "rmse", "value"])

    return df


def make_experiment_df_kl(experiments, ward_names, column_names, epsilons, mus):
    data = []
    for name in ward_names:

        val_list = []
        for i in range(len(experiments)):
            metrics = experiments[i].query("area_name == @name").copy(deep=True)
            for epsilon in epsilons:
                for mu in mus:
                    for column_name in column_names:
                        val_list.append(
                            (
                                name,
                                epsilon,
                                mu,
                                column_name,
                                metrics.loc[(epsilon, mu), column_name].tolist(),
                            )
                        )
        data.extend(val_list)

    df = pd.DataFrame(
        data=data, columns=["area_name", "epsilon", "mu", "kl_divergence", "value"]
    )
    return df


def make_experiment_df_merged(
    experiments_clip, experiments_no_clip, ward_names, column_names, epsilons
):
    data = []
    for name in ward_names:
        val_list = []

        for i in range(len(experiments_clip)):
            metrics_clip = (
                experiments_clip[i]["metrics_df"]
                .query("area_name == @name")
                .copy(deep=True)
            )
            metrics_no_clip = (
                experiments_no_clip[i]["metrics_df"]
                .query("area_name == @name")
                .copy(deep=True)
            )
            for epsilon in epsilons:
                for column_name in column_names:
                    val_list.append(
                        (
                            name,
                            epsilon,
                            column_name,
                            metrics_clip.loc[epsilon][column_name].tolist(),
                            "Clipping",
                        )
                    )
                    val_list.append(
                        (
                            name,
                            epsilon,
                            column_name,
                            metrics_no_clip.loc[epsilon][column_name].tolist(),
                            "No Post-Processing",
                        )
                    )

        data.extend(val_list)

    df = pd.DataFrame(
        data=data, columns=["area_name", "epsilon", "rmse", "value", "postprocessing"]
    )

    return df


def make_experiment_df_merged_laplace(
    experiments_clip,
    experiments_round,
    experiments_clip_round,
    experiments,
    ward_names,
    column_names,
    epsilons,
):
    data = []
    for name in ward_names:
        val_list = []

        for i in range(len(experiments_clip)):
            metrics_clip = (
                experiments_clip[i]["metrics_df"]
                .query("area_name == @name")
                .copy(deep=True)
            )
            metrics_round = (
                experiments_round[i]["metrics_df"]
                .query("area_name == @name")
                .copy(deep=True)
            )
            metrics_clip_round = (
                experiments_clip_round[i]["metrics_df"]
                .query("area_name == @name")
                .copy(deep=True)
            )
            metrics_no_clip = (
                experiments[i]["metrics_df"].query("area_name == @name").copy(deep=True)
            )

            for epsilon in epsilons:
                for column_name in column_names:
                    val_list.append(
                        (
                            name,
                            epsilon,
                            column_name,
                            metrics_clip.loc[epsilon][column_name].tolist(),
                            "Clipping",
                        )
                    )
                    val_list.append(
                        (
                            name,
                            epsilon,
                            column_name,
                            metrics_clip_round.loc[epsilon][column_name].tolist(),
                            "Clipping and Rounding",
                        )
                    )
                    val_list.append(
                        (
                            name,
                            epsilon,
                            column_name,
                            metrics_round.loc[epsilon][column_name].tolist(),
                            "Rounding",
                        )
                    )
                    val_list.append(
                        (
                            name,
                            epsilon,
                            column_name,
                            metrics_no_clip.loc[epsilon][column_name].tolist(),
                            "No Post-Processing",
                        )
                    )

        data.extend(val_list)

    df = pd.DataFrame(
        data=data, columns=["area_name", "epsilon", "rmse", "value", "postprocessing"]
    )

    return df


def make_experiment_df_populations(
    experiments,
    ward_codes,
    column_names_pop,
    column_names_pop_dp,
    column_names,
    column_names_dp,
):
    list_wards = []
    for code in ward_codes:

        df_list = []
        for i in range(len(experiments)):
            ward = experiments[i]["wards_dp"][code][0].copy(deep=True)
            area_name = experiments[i]["wards_dp"][code][1]["area_name"]

            df = ward[column_names_pop + column_names + ["EthnicGroup"]]

            df[column_names_pop] = df[column_names_pop] * 100
            df_wide = pd.wide_to_long(
                df,
                column_names_pop_dp + column_names_dp,
                i="EthnicGroup",
                j="epsilon",
                sep=" ",
                suffix=r"[0-9]+\.?([0-9]+)?",
            )
            df_wide.reset_index(inplace=True)
            df_wide["area_name"] = area_name

            df_list.append(df_wide)

        list_wards.extend(df_list)

    dfs = pd.concat(list_wards, axis=0, ignore_index=True, sort=False)

    dfs["median"] = dfs.groupby(["EthnicGroup", "epsilon", "area_name"])[
        "dp %"
    ].transform("median")
    dfs["mean"] = dfs.groupby(["EthnicGroup", "epsilon", "area_name"])[
        "dp %"
    ].transform("mean")
    dfs["inc_frac"] = (
        dfs.groupby(["EthnicGroup", "epsilon", "area_name"])[
            "significantly_increased"
        ].transform("sum")
        / dfs.groupby(["EthnicGroup", "epsilon", "area_name"])[
            "significantly_increased"
        ].transform("size")
    ) * 100
    dfs["dec_frac"] = (
        dfs.groupby(["EthnicGroup", "epsilon", "area_name"])[
            "significantly_decreased"
        ].transform("sum")
        / dfs.groupby(["EthnicGroup", "epsilon", "area_name"])[
            "significantly_decreased"
        ].transform("size")
    ) * 100

    return dfs
