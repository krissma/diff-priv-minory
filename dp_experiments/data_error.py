import numpy as np
import evaluation_helpers
import os


def apply_data_error(cl_width_rel, value):
    # z-score for a 95% confidence interval
    z_score = 1.96
    generator = np.random.default_rng()
    cl_abs = (value * cl_width_rel) * 2
    std = cl_abs / z_score
    noised_value = generator.normal(value, std)
    return round(noised_value)


# creates a new column with alternate data in an existing data frame
# negative numbers are rounded to zero and numbers are rounded to next integer
def create_data_error_column(df, column, column_name, cl_width_rel):
    df[column_name] = df.apply(
        lambda df: apply_data_error(cl_width_rel, df[column]), axis=1
    )
    return df


def apply_data_error_to_dataframe(csv, sheet_cl, path, table_name, geo_lookup, ward):
    ward_codes = csv["GeographyCode"].tolist()
    csv_copy = csv.copy(deep=True)
    csv_copy.set_index("GeographyCode", inplace=True)
    for code in ward_codes:
        la_code = ward.return_la(code, geo_lookup)
        cl_width_rel = evaluation_helpers.get_la_cl(la_code, sheet_cl)
        csv_copy.loc[code] = csv_copy.apply(
            lambda df: apply_data_error(cl_width_rel, df.loc[code]), axis=0
        )
    csv_copy.reset_index(inplace=True)
    csv_copy.to_csv(os.path.join(path, f"{table_name}_data_error.csv"), index=False)

    return None


def apply_data_error_to_dataframe_la(csv, sheet_cl, path, table_name):
    la_codes = csv["GeographyCode"].tolist()
    csv_copy = csv.copy(deep=True)
    csv_copy.set_index("GeographyCode", inplace=True)
    for code in la_codes:
        cl_width_rel = evaluation_helpers.get_la_cl(code, sheet_cl)
        csv_copy.loc[code] = csv_copy.apply(
            lambda df: apply_data_error(cl_width_rel, df.loc[code]), axis=0
        )
    csv_copy.reset_index(inplace=True)
    csv_copy.to_csv(os.path.join(path, f"{table_name}_data_error.csv"), index=False)

    return None
