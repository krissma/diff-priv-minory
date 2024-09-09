import pandas as pd

# Different dataframes have different structures. These different structures are handled with a factory pattern

DF_CREATORS = {}


def register_df_creators(df_type):
    def decorator(fn):
        DF_CREATORS[df_type] = fn
        return fn

    return decorator


# * function for creating a df without nested categories
@register_df_creators("normal")
def create_df(sheet, column_names, table_name):
    sheet_copy = sheet.copy(deep=True)

    # creating the names for the rows
    rows = sheet_copy.index.tolist()

    # creating the names for the columns
    columns = sheet_copy.columns.values.tolist()

    # Creating row values. The values identify a combination of different categories in the csv sheet.
    row_values = []
    for i, row in enumerate(rows):
        row_values.append(sheet_copy.loc[row].tolist())

    # creating an ordered list of dataset columns
    values_list = []

    for i, row in enumerate(row_values):
        for j, value in enumerate(row):
            sublist = []
            sublist.append(rows[i])
            sublist.append(columns[j])
            sublist.append(table_name + str(row[j]).zfill(4))
            values_list.append(sublist)

    df = pd.DataFrame(data=values_list, columns=column_names)
    df.drop_duplicates(inplace=True, ignore_index=True)
    df.sort_values(by="Dataset", inplace=True)
    return df


# * function for creating a df with nested categories
@register_df_creators("nested")
def create_df_nested(sheet, column_names, table_name, num_nested_category):

    # creating the names for the rows
    sheet_copy = sheet.copy(deep=True)
    rows = sheet_copy.index.tolist()

    # creating the names for the columns
    columns = sheet_copy.columns.values.tolist()

    # extracting the additional row category
    rows_categories = rows[:: num_nested_category + 1]

    # removing the nested values from the row names
    rows = [i for j, i in enumerate(rows) if j % (num_nested_category + 1) != 0]

    # now that I have read in the additional row categories, I can drop the NaN values
    sheet_copy.dropna(inplace=True, ignore_index=True)

    column_values = []
    for column in columns:
        column_values.append(sheet_copy.loc[:, column].tolist())

    # creating list of categories
    cat_list = []
    for column in column_values:
        sublist = []
        for i in range(len(rows_categories)):
            sublist.extend([rows_categories[i]] * num_nested_category)
        cat_list.append(sublist)

    values_list = []

    for j, column in enumerate(column_values):
        for i, val in enumerate(column):
            sublist = []
            sublist.append(cat_list[j][i])
            sublist.append(rows[i])
            sublist.append(columns[j])
            sublist.append(table_name + str(val).zfill(4))
            values_list.append(sublist)

    df = pd.DataFrame(data=values_list, columns=column_names)
    df.sort_values(by="Dataset", inplace=True)
    return df


def get_df_creator(df_type):
    return DF_CREATORS.get(df_type)
