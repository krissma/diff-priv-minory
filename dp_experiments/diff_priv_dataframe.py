import os
import diff_priv


def apply_laplace_to_dataframe(
    path,
    csv,
    datasets,
    ward,
    geo_lookup,
    table_name,
    sensitivity,
    epsilons,
    clipping,
    rounding,
    random_state=None,
):

    for epsilon in epsilons:
        csv_dp = diff_priv.apply_laplace(
            csv,
            datasets,
            epsilon=epsilon,
            delta=0,
            sensitivity=sensitivity,
            clipping=clipping,
            rounding=rounding,
        )
        df_dp = ward.get_ward(csv_dp, geo_lookup)
        df_dp.to_csv(os.path.join(path, f"{table_name}_{epsilon}_dp.csv"), index=False)

    return None


def apply_laplace_to_dataframe_la(
    path,
    csv,
    datasets,
    local_authority,
    geo_lookup,
    table_name,
    sensitivity,
    epsilons,
    clipping,
    rounding,
    random_state=None,
):

    for epsilon in epsilons:
        csv_dp = diff_priv.apply_laplace(
            csv,
            datasets,
            epsilon=epsilon,
            delta=0,
            sensitivity=sensitivity,
            clipping=clipping,
            rounding=rounding,
            random_state=None,
        )
        df_dp = local_authority.get_local_authority(csv_dp, geo_lookup)
        df_dp.to_csv(os.path.join(path, f"{table_name}_{epsilon}_dp.csv"), index=False)

    return None


def apply_laplace_to_dataframe_seed(
    path,
    csv,
    ward,
    geo_lookup,
    table_name,
    sensitivity,
    delta,
    epsilons,
    clipping,
    rounding,
):
    table_num = "".join(filter(str.isdigit, table_name))
    for epsilon in epsilons:
        csv_copy = csv.copy(deep=True)
        csv_copy.set_index("GeographyCode", inplace=True)
        csv_dp = csv_copy.apply(
            lambda row: diff_priv.laplace_seed(
                row,
                (int(row.name[1:3] + row.name[5:] + table_num)),
                epsilon,
                delta,
                sensitivity,
                clipping,
                rounding,
            ),
            axis=1,
        )

        df_dp = ward.get_ward(csv_dp, geo_lookup)
        df_dp.to_csv(os.path.join(path, f"{table_name}_{epsilon}_dp.csv"), index=False)

    return df_dp


def apply_laplace_to_dataframe_seed_la(
    path,
    csv,
    local_authority,
    geo_lookup,
    table_name,
    sensitivity,
    delta,
    epsilons,
    clipping,
    rounding,
):
    table_num = "".join(filter(str.isdigit, table_name))
    for epsilon in epsilons:
        csv_copy = csv.copy(deep=True)
        csv_copy.set_index("GeographyCode", inplace=True)
        csv_dp = csv_copy.apply(
            lambda row: diff_priv.laplace_seed(
                row,
                (int(row.name[1:3] + row.name[5:] + table_num)),
                epsilon,
                delta,
                sensitivity,
                clipping,
                rounding,
            ),
            axis=1,
        )

        df_dp = local_authority.get_local_authority(csv_dp, geo_lookup)
        df_dp.to_csv(os.path.join(path, f"{table_name}_{epsilon}_dp.csv"), index=False)

    return df_dp


def apply_geometric_to_dataframe(
    path,
    csv,
    datasets,
    ward,
    geo_lookup,
    table_name,
    sensitivity,
    epsilons,
    clipping,
    random_state=None,
):

    for epsilon in epsilons:
        csv_dp = diff_priv.apply_geometric(
            csv, datasets, epsilon=epsilon, sensitivity=sensitivity, clipping=clipping
        )
        df_dp = ward.get_ward(csv_dp, geo_lookup)
        df_dp.to_csv(os.path.join(path, f"{table_name}_{epsilon}_dp.csv"), index=False)

    return None


def apply_geometric_to_dataframe_la(
    path,
    csv,
    datasets,
    local_authority,
    geo_lookup,
    table_name,
    sensitivity,
    epsilons,
    clipping,
    random_state=None,
):

    for epsilon in epsilons:
        csv_dp = diff_priv.apply_geometric(
            csv, datasets, epsilon=epsilon, sensitivity=sensitivity, clipping=clipping
        )
        df_dp = local_authority.get_local_authority(csv_dp, geo_lookup)
        df_dp.to_csv(os.path.join(path, f"{table_name}_{epsilon}_dp.csv"), index=False)

    return None


def apply_geometric_to_dataframe_seed(
    path, csv, ward, geo_lookup, table_name, sensitivity, epsilons, clipping
):
    table_num = "".join(filter(str.isdigit, table_name))
    for epsilon in epsilons:
        csv_copy = csv.copy(deep=True)
        csv_copy.set_index("GeographyCode", inplace=True)
        csv_dp = csv_copy.apply(
            lambda row: diff_priv.geometric_seed(
                row,
                (int(row.name[1:3] + row.name[5:] + table_num)),
                epsilon,
                sensitivity,
                clipping,
            ),
            axis=1,
        )

        df_dp = ward.get_ward(csv_dp, geo_lookup)
        df_dp.to_csv(os.path.join(path, f"{table_name}_{epsilon}_dp.csv"), index=False)

    return df_dp


def apply_geometric_to_dataframe_seed_la(
    path, csv, local_authority, geo_lookup, table_name, sensitivity, epsilons, clipping
):
    table_num = "".join(filter(str.isdigit, table_name))
    for epsilon in epsilons:
        csv_copy = csv.copy(deep=True)
        csv_copy.set_index("GeographyCode", inplace=True)
        csv_dp = csv_copy.apply(
            lambda row: diff_priv.geometric_seed(
                row,
                (int(row.name[1:3] + row.name[5:] + table_num)),
                epsilon,
                sensitivity,
                clipping,
            ),
            axis=1,
        )

        df_dp = local_authority.get_local_authority(csv_dp, geo_lookup)
        df_dp.to_csv(os.path.join(path, f"{table_name}_{epsilon}_dp.csv"), index=False)

    return df_dp
