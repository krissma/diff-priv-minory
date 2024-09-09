import pandas as pd
import os
from functools import reduce

DATA_DIR = "../census_data"


class Ward:

    def __init__(self):
        pass

    # load geo lookup file
    def get_geo_lookup_ward(self):
        # cmwd stands for Census merged ward, lad stands for Local authority district
        df = pd.read_csv(
            os.path.join(
                DATA_DIR,
                "geofiles",
                "Census_Merged_Wards_Dec_2011_FEB_in_England_and_Wales_2022_5708607548530513025.csv",
            ),
            usecols=["cmwd11cd", "cmwd11nm", "lad11cd", "lad11nm"],
            encoding="ISO-8859-1",
            dtype=str,
        )
        df.drop_duplicates(inplace=True, ignore_index=True)
        df.rename(columns=str.upper, inplace=True)
        return df

    # * function merging ward csv with geodata
    def get_ward(self, df, geo_lookup_df):
        # filtering for a certain area, for wards the GeographyCode is the census merged wards column
        geo_lookup = geo_lookup_df.copy(deep=True)
        geo_lookup.rename(
            columns={
                "CMWD11CD": "GeographyCode",
                "CMWD11NM": "Name",
                "LAD11CD": "GeographyCodeLA",
                "LAD11NM": "NameLA",
            },
            inplace=True,
        )
        merged_frame = pd.merge(geo_lookup, df, how="inner", on=["GeographyCode"])
        return merged_frame

    # * function filtering for wards
    def filter_ward(self, df, geography_code):
        df_ward = df[df["GeographyCode"].str.startswith(geography_code)]
        df_ward.drop_duplicates(inplace=True, ignore_index=True)
        return df_ward

    # * function to look up features for a specific ward
    def lookup_features_ward(self, df, datasets):
        column_list = ["GeographyCode", "Name", "GeographyCodeLA", "NameLA"]
        column_list.extend(datasets)

        df_london_features = df[column_list]
        return df_london_features

    # * function that returns a dataframe with population numbers for one specific area for a ward
    def get_population_ward(self, df_area, lookup_df):
        population_numbers = df_area.loc[0].values[4:]
        df = lookup_df.copy(deep=True)
        df["PopulationNumbers"] = population_numbers
        return df

    def get_total_population_ward(self, df, area_code, table_name):
        df_area = self.filter_ward(df, area_code)
        return df_area[table_name + "0001"][0]

    # returns the code of the local area of a ward
    def return_la(self, area_code, df_area):
        la = df_area[df_area["CMWD11CD"].str.startswith(area_code)]
        return la["LAD11CD"].values[0]


class LocalAuthority:

    def __init__(self):
        pass

    def get_geo_lookup_LA(self):
        # cmwd stands for Census merged ward, lad stands for Local authority district
        df = pd.read_csv(
            os.path.join(
                DATA_DIR,
                "geofiles",
                "Census_Merged_Wards_Dec_2011_FEB_in_England_and_Wales_2022_5708607548530513025.csv",
            ),
            usecols=["cmwd11cd", "cmwd11nm", "lad11cd", "lad11nm"],
            encoding="ISO-8859-1",
            dtype=str,
        )

        df.drop_duplicates(inplace=True, ignore_index=True)
        df.rename(columns=str.upper, inplace=True)
        return df

    # * function merging local area csv with geodata
    def get_local_authority(self, df, geo_lookup_df):
        # filtering for csvs at ward level
        geo_lookup = geo_lookup_df.copy(deep=True)
        # for levels 4 and 5 the GeographyCode is the local area code
        geo_lookup.rename(
            columns={"LAD11CD": "GeographyCode", "LAD11NM": "NameLA"}, inplace=True
        )
        merged_frame = pd.merge(
            geo_lookup[["GeographyCode", "NameLA"]],
            df,
            how="inner",
            on=["GeographyCode"],
        )
        merged_frame.drop_duplicates(inplace=True, ignore_index=True)
        return merged_frame

    # * function filtering for local authority, possible levels are 4,5 or 6, where 6 is the ward level
    def filter_local_authority(self, df, geography_code, level):
        # filtering for csvs at ward level
        if level == 6:
            # filtering for a certain area, for wards the GeographyCode is the census merged wards column
            df_region = df[df["GeographyCodeLA"].str.startswith(geography_code)]
            df_region.drop_duplicates(inplace=True, ignore_index=True)
        else:
            # for levels 4 and 5 the GeographyCode is the local area code
            df_region = df[df["GeographyCode"].str.startswith(geography_code)]
            df_region.drop_duplicates(inplace=True, ignore_index=True)
        return df_region

    # * function to look up features for a specific local authority
    def lookup_features_local_authority(self, df, datasets):
        column_list = ["GeographyCode", "NameLA"]
        column_list.extend(datasets)

        df_london_features = df[column_list]
        return df_london_features

    # * function that returns a dataframe with population numbers for one specific area for local authorities or regions
    def get_population_local_authority(self, df_area, lookup_df):
        population_numbers = df_area.loc[0].values[2:]
        df = lookup_df.copy(deep=True)
        df["PopulationNumbers"] = population_numbers
        return df

    def get_total_population_la(self, df, area_code, table_name):
        df_area = self.filter_local_authority(df, area_code, 5)
        return df_area[table_name + "0001"][0]


class Region:

    def __init__(self):
        pass

    def get_geo_lookup_region(self):
        df = pd.read_csv(
            os.path.join(
                DATA_DIR,
                "geofiles",
                "Middle_Layer_Super_Output_Area_(2011)_to_Built-up_Area_Sub_Division_to_Built-up_Area_to_Local_Authority_District_to_Region_(December_2011)_Lookup_in_England_and_Wales.csv",
            ),
            usecols=["RGN11CD", "RGN11NM", "LAD11CD", "LAD11NM"],
            dtype=str,
        )
        df.drop_duplicates(inplace=True, ignore_index=True)
        return df

    # * function returning a dataframe with the respective region codes and names
    def get_region(self, df, geo_lookup_df):
        geo_lookup_df.rename(
            columns={"RGN11CD": "GeographyCode", "RGN11NM": "NameRegion"}, inplace=True
        )
        filtered_frame = pd.merge(geo_lookup_df, df, how="inner", on=["GeographyCode"])
        return filtered_frame

    # * function filtering for region
    def filter_region(self, df, geography_code):
        df_region = df[df["GeographyCode"].str.startswith(geography_code)]
        df_region.drop_duplicates(inplace=True, ignore_index=True)
        return df_region

    # * function that returns a dataframe with population numbers for one specific area for local authorities or regions
    def get_population_region(self, df_area, lookup_df):
        population_numbers = df_area.loc[0].values[2:]
        df = lookup_df.copy(deep=True)
        df["PopulationNumbers"] = population_numbers
        return df

    def get_total_population_region(self, df, area_code, table_name):
        df_area = self.filter_region(df, area_code)
        return df_area[table_name + "0001"][0]
