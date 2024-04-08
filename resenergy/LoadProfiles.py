import pandas as pd
import pytz
import os
from enum import Enum, auto

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Data file path
base_dir = os.path.join(current_dir, '..', 'data', 'LoadProfiles')


class Profiles(Enum):
    # Multifamily with natural gas heating and no A/C
    HotDryScenario1 = auto()
    # MF with nat gas heating and A/C
    HotDryScenario2 = auto()
    # MF with heat pump heating and cooling
    HotDryScenario3 = auto()
    # MF with gas water heater, HP heating and cooling
    HotDryScenario4 = auto()
    # MF with electric water heating, HP heating and cooling
    HotDryScenario5 = auto()
    # MF with electric water heating, HP heating and cooling, gas range
    HotDryScenario6 = auto()
    # MF with electric water heating, HP heating and cooling, electric range
    HotDryScenario7 = auto()
    # Base house loads:
    HotDryBaseHouse = auto()
    MarineBaseHouse = auto()


def get_multifamily_building_load_profile(
        load_csv_path=None,
        house_load=True,
        house_load_annual_total=10000,
        units_load=False,
        units_load_num=10,
):
    if load_csv_path not in (None, ''):
        # TODO: call CSV file loader
        raise NotImplementedError
        # When implemented, return directly

    if house_load:
        house_load_profile_df = load_house_loadprof(Profiles.HotDryBaseHouse, house_load_annual_total)

    if units_load:
        units_load_profile_df = load_units_loadprof(Profiles.HotDryScenario2, units_load_num)
        if house_load:
            # Merge the two dataframes
            load_profile_df = house_load_profile_df + units_load_profile_df
        else:
            load_profile_df = units_load_profile_df
    else:
        load_profile_df = house_load_profile_df
    return load_profile_df


def load_house_loadprof(scenario, total_annual, tzname='US/Pacific'):
    """
    Load the house load profile for a given scenario
    :param scenario: From Profiles Enum
    :param tzname: pytz timezone name
    :param total_annual: Scales the default load profile to this total annual energy consumption
    :return:
    """
    if scenario == Profiles.HotDryBaseHouse:
        fname = 'BaselineElectricProfile_HotDry.csv'
    elif scenario == Profiles.MarineBaseHouse:
        fname = 'BaselineElectricProfile_Marine.csv'
    else:
        fname = 'BaselineElectricProfile_General.csv'

    fpath = os.path.join(base_dir, fname)
    df = pd.read_csv(fpath)

    # Convert string to datetime object
    df['Timestamp (EST)'] = pd.to_datetime(df['Timestamp (EST)'])
    # Right now, just going to keep electricity profile
    load_profile_df = df[['Timestamp (EST)',
                    'baseline.out.electricity.total.energy_consumption.kwh'
                    ]]
    rename_cols_dict = {
        'Timestamp (EST)': 'Timestamp',
        'baseline.out.electricity.total.energy_consumption.kwh': 'baseline_total'
    }
    load_profile_df = load_profile_df.rename(columns=rename_cols_dict)

    # Change to relevant timezone
    load_profile_df = load_profile_df.set_index('Timestamp')
    load_profile_df = load_profile_df.tz_convert(pytz.timezone(tzname))

    # Normal and set annual total
    scale_factor = total_annual / load_profile_df['baseline_total'].sum()
    load_profile_df['baseline_total'] = load_profile_df['baseline_total'] * scale_factor

    # For convenience, wrap around so the whole datetimeindex is in 2018.
    index2017 = load_profile_df[load_profile_df.index.year == 2017]  # get any 2017 data
    index2017.index = index2017.index + pd.DateOffset(years=1)  # roll over to 2018
    load_profile_df = pd.concat([load_profile_df, index2017])  # concatenate with prior data
    load_profile_df = load_profile_df[load_profile_df.index.year == 2018]  # keep only 2017

    return load_profile_df


def load_units_loadprof(scenario=Profiles.HotDryScenario2, n_units=1, tzname='US/Pacific'):
    if scenario == Profiles.HotDryScenario1:
        fname = 'resstock_dry_hot_s1.csv'
        total_units_repr = 2030  # 2180
        weight = 242.131  # another column in the metadata for normalizing
        expected_total_elec = 3204  # from the CSV, look at the total use, averaged against scenarios.
        # Used as a check that the exported data matches the expected filters and number of cases.
    elif scenario == Profiles.HotDryScenario2:
        fname = 'resstock_dry_hot_s2.csv'
        total_units_repr = 4185
        weight = 242.131  # another column in the metadata for normalizing
        expected_total_elec = 5394.1
    elif scenario == Profiles.HotDryScenario3:
        fname = 'resstock_dry_hot_s3.csv'
        total_units_repr = 848
        weight = 242.131  # another column in the metadata for normalizing
        expected_total_elec = 6288
    elif scenario == Profiles.HotDryScenario4:
        fname = 'resstock_dry_hot_s4.csv'
        total_units_repr = 521
        weight = 242.131  # another column in the metadata for normalizing
        expected_total_elec = 5892
    elif scenario == Profiles.HotDryScenario5:
        fname = 'resstock_dry_hot_s5.csv'
        total_units_repr = 258
        weight = 242.131  # another column in the metadata for normalizing
        expected_total_elec = 7182
    elif scenario == Profiles.HotDryScenario6:
        fname = 'resstock_dry_hot_s6.csv'
        total_units_repr = 73
        weight = 242.131  # another column in the metadata for normalizing
        expected_total_elec = 6577
    elif scenario == Profiles.HotDryScenario7:
        fname = 'resstock_dry_hot_s7.csv'
        total_units_repr = 59
        weight = 242.131  # another column in the metadata for normalizing
        expected_total_elec = 7555
    else:
        raise Exception

    fpath = os.path.join(base_dir, fname)
    df = pd.read_csv(fpath)

    # Convert string to datetime object
    df['Timestamp (EST)'] = pd.to_datetime(df['Timestamp (EST)'])
    # Right now, just going to keep electricity profile
    load_profile_df = df[['Timestamp (EST)',
                          'baseline.out.electricity.total.energy_consumption.kwh'
                          ]]
    rename_cols_dict = {
        'Timestamp (EST)': 'Timestamp',
        'baseline.out.electricity.total.energy_consumption.kwh': 'baseline_total'
    }
    load_profile_df = load_profile_df.rename(columns=rename_cols_dict)

    # Change to relevant timezone
    load_profile_df = load_profile_df.set_index('Timestamp')
    load_profile_df = load_profile_df.tz_convert(pytz.timezone(tzname))

    # Scale by number of units represented
    load_profile_df = load_profile_df / total_units_repr / weight
    total_elec = load_profile_df['baseline_total'].sum()
    if abs(total_elec - expected_total_elec)/total_elec > 0.01:
        print("!!! WARNING: Timeseries data is likely off!! ")
    load_profile_df *= n_units

    # For convenience, wrap around so the whole datetimeindex is in 2018.
    index2017 = load_profile_df[load_profile_df.index.year == 2017]  # get any 2017 data
    index2017.index = index2017.index + pd.DateOffset(years=1)  # roll over to 2018
    load_profile_df = pd.concat([load_profile_df, index2017])  # concatenate with prior data
    load_profile_df = load_profile_df[load_profile_df.index.year == 2018]  # keep only 2017

    return load_profile_df

