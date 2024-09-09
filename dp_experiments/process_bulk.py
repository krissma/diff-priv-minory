import pandas as pd
import os
from functools import reduce
import df_creator

DATA_DIR = "../census_data"


class ProcessBulk:

    def __init__(self, bulk_folder, index_sheet):
        self.bulk_folder = bulk_folder
        self.index_sheet = index_sheet
        self.bulk_data = os.path.join(DATA_DIR, self.bulk_folder)

    def set_bulk_folder(self):
        self.bulk_data = os.path.join(DATA_DIR, self.bulk_folder)

    def get_bulk_data_path(self):
        return self.bulk_data

    # this function reads in the excel index sheet which shows which topics are available in the bulk dataset
    def get_index(self):
        f
        index_file = os.path.join(self.bulk_data, self.index_sheet)
        f = pd.ExcelFile(index_file)
        index = f.parse(sheet_name="Index")
        return index

    # * Excel index files for different bulk data sets have different formatting, therefore the amount of rows that need to be skipped varies
    # * the default skip value is 6
    def read_table(self, table_name, skip_row=6):
        # the value in the last row is copyright information, the value in the row before is nan, therefore I drop the last two rows
        sheet = pd.read_excel(
            os.path.join(self.bulk_data, self.index_sheet),
            sheet_name=table_name,
            skiprows=skip_row,
            header=None,
            skipfooter=2,
        )
        # setting the column headers
        sheet.columns = sheet.iloc[0]
        sheet = sheet[1:]
        sheet = sheet.reset_index(drop=True)
        sheet.rename(columns={sheet.columns[0]: "Categories"}, inplace=True)
        # setting row index
        sheet = sheet.set_index("Categories")
        return sheet

    def read_cl(self, table_name):
        sheet_cl = pd.read_excel(
            os.path.join(self.bulk_data, self.index_sheet),
            sheet_name=table_name,
            skiprows=5,
            header=None,
            skipfooter=17,
        )
        # setting the column headers
        sheet_cl.columns = sheet_cl.iloc[0]
        sheet_cl = sheet_cl[1:]
        sheet_cl = sheet_cl.reset_index(drop=True)
        sheet_cl.rename(
            columns={
                "Area code 1": "area_code",
                "Area name": "area_name",
                "Relative confidence interval width 2": "relative_cl_width",
            },
            inplace=True,
        )
        # setting row index
        sheet_cl.dropna(how="all", inplace=True)
        sheet_cl.dropna(how="all", axis="columns", inplace=True)
        return sheet_cl

    # * Excel index files for different bulk data sets have different formatting, therefore the amount of rows that need to be skipped varies
    # * the default skip value is 6
    # * here only part of the table is read, e.g. if the table has more sub-categories than are needed
    # * nrows describes the amount of rows that are read, start_sheet describes where the actual sheet that is read starts after setting the column headers
    def read_sub_table(self, table_name, nrows, start_sheet=2, skip_row=6):
        # the value in the last row is copyright information, the value in the row before is nan, therefore I drop the last two rows
        sheet = pd.read_excel(
            os.path.join(self.bulk_data, self.index_sheet),
            sheet_name=table_name,
            skiprows=skip_row,
            nrows=nrows,
            header=None,
        )
        # setting the column headers
        sheet.columns = sheet.iloc[0]
        sheet = sheet[start_sheet:]
        sheet = sheet.reset_index(drop=True)
        sheet.rename(columns={sheet.columns[0]: "Categories"}, inplace=True)
        # setting row index
        sheet = sheet.set_index("Categories")
        return sheet

    # * this function reads in the csvs for a table name. This should also work when there is only one csv file which is not split

    # * geographical level can be set with a number, where for wards:
    # * 1. Highest geography level, England and Wales combined.
    # * 2. England, Wales.
    # * 3. Regions and Wales.
    # * 4. Unitary Authorities, Counties, Metropolitan Counties, Inner and Outer London, Welsh Unitary Authorities.
    # * 5. Unitary Authorities, Non Metropolitan District, Metropolitan Districts, London Boroughs, Welsh Unitary Authorities.
    # * 6. Electoral Wards /Divisions and Welsh Divisions

    #! currently only implementing 4-6
    def read_csv(self, table_name, level, subfolder=""):
        # csv files might be in a subfolder, subfolder is given as an optional argument here
        path_to_csv = os.path.join(self.bulk_data, subfolder)

        # get a list of all csv files that need to be read
        csv_files = [
            f
            for f in os.listdir(path_to_csv)
            if (f.startswith(table_name + "DATA") and f.endswith(str(level) + ".CSV"))
        ]
        csv_files.sort()

        # now the csv files are read into a dataframe and combined
        dfs = []
        for f in csv_files:
            df = pd.read_csv(os.path.join(path_to_csv, f))
            dfs.append(df)

        final_df = reduce(
            lambda left, right: pd.merge(
                left, right, on=["GeographyCode"], how="outer"
            ),
            dfs,
        )
        final_df.drop_duplicates(inplace=True, ignore_index=True)
        return final_df

        #! currently only implementing 4-6

    def read_csv_sub_table(self, ncols, table_name, level, subfolder=""):
        # csv files might be in a subfolder, subfolder is given as an optional argument here
        path_to_csv = os.path.join(self.bulk_data, subfolder)
        print(path_to_csv)

        # get a list of all csv files that need to be read
        csv_files = [
            f
            for f in os.listdir(path_to_csv)
            if (f.startswith(table_name + "DATA") and f.endswith(str(level) + ".CSV"))
        ]
        csv_files.sort()

        # now the csv files are read into a dataframe and combined
        dfs = []
        for f in csv_files:
            df = pd.read_csv(os.path.join(path_to_csv, f))
            dfs.append(df)

        final_df = reduce(
            lambda left, right: pd.merge(
                left, right, on=["GeographyCode"], how="outer"
            ),
            dfs,
        )
        final_df = final_df.iloc[:, : ncols + 1]
        final_df.drop_duplicates(inplace=True, ignore_index=True)
        return final_df

    # function to set up lookup_df and read in csvs
    def set_up(
        self, table_name, df_type, column_names, num_nested_category, subfolder, level
    ):
        dfs = dict()
        dfs["sheet"] = self.read_table(table_name=table_name, skip_row=4)
        dfs["lookup_df"] = df_creator.get_df_creator(df_type)(
            dfs["sheet"],
            column_names,
            table_name,
            num_nested_category=num_nested_category,
        )
        dfs["csv_df"] = self.read_csv(
            table_name=table_name, level=level, subfolder=subfolder
        )
        return dfs

    # function to set up lookup_df and read in csvs
    def set_up_sub_table(
        self,
        table_name,
        df_type,
        column_names,
        num_nested_category,
        nrows,
        start_sheet,
        subfolder,
        level,
    ):
        dfs = dict()
        dfs["sheet"] = self.read_sub_table(
            table_name, nrows=nrows, start_sheet=start_sheet, skip_row=4
        )
        dfs["lookup_df"] = df_creator.get_df_creator(df_type)(
            dfs["sheet"],
            column_names,
            table_name,
            num_nested_category=num_nested_category,
        )
        # since only a sub table is read in, the csvs have to be cropped to only include the subcategories
        index = dfs["lookup_df"].index
        print(len(index))
        dfs["csv_df"] = self.read_csv_sub_table(
            (len(index)), table_name=table_name, level=level, subfolder=subfolder
        )
        return dfs

    # * function to get a combination of features
    def get_reduced_features(self, df, filter_dict):
        # if this function is called by the get_filtered_df function, the dictionary might be empty
        if filter_dict is not None:
            reduced_lookup_df = df.loc[
                df[list(filter_dict)].isin(filter_dict).all(axis=1), :
            ]
            return reduced_lookup_df
        else:
            return df

    # * returns a list of column names for the datasets in the lookup dataframe
    def get_list_of_datasets(self, df):
        filtered_df = df["Dataset"].values
        datasets = []
        for val in filtered_df:
            datasets.append(val)
        return datasets

    # * function to retrieve the list of features for a specific dataset
    def get_feature_list(self, df, dataset_num):
        features = (
            df.loc[df["Dataset"].str.endswith(dataset_num[-4:])].values[0].tolist()
        )
        return features[:-1]

    # * this function combines several filtering steps for a region
    def get_filtered_df_region(self, region, df, area_code, lookup_df, filter_dict):
        df_area = region.filter_region(df, area_code)
        df_pop = region.get_population_region(df_area, lookup_df)
        df_feature = self.get_reduced_features(df_pop, filter_dict)
        return df_feature

    # * this function combines several filtering steps for a local authority
    def get_filtered_df_la(self, la, df, area_code, lookup_df, filter_dict):
        # I am using level 5 as a default here

        df_area = la.filter_local_authority(df, area_code, 5)
        area_name = df_area.iloc[:, 1].values[0]
        df_pop = la.get_population_local_authority(df_area, lookup_df)
        df_feature = self.get_reduced_features(df_pop, filter_dict)

        return df_feature, area_name

    # * this function combines several filtering steps for a ward
    def get_filtered_df_ward(self, ward, df, area_code, lookup_df, filter_dict):
        df_area = ward.filter_ward(df, area_code)
        area_name = df_area.iloc[:, 1].values[0]
        df_pop = ward.get_population_ward(df_area, lookup_df)
        df_feature = self.get_reduced_features(df_pop, filter_dict)

        return df_feature, area_name
