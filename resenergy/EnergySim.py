import pandas as pd

from resenergy.LoadProfiles import get_multifamily_building_load_profile
from resenergy.Solar import SolarSystem
from resenergy.Utility import Utility


def run_energy_sim(
        utilityParams,
        baselineLoadProfileParams,
        solarSystemParams=None,
        batteryParams=None
):

    # Load the baseline load profile
    building_df = get_multifamily_building_load_profile(**baselineLoadProfileParams)
    building_df['total_load'] = building_df['baseline_total']

    # Create a Utility instance using utilityParams
    utility = Utility(**utilityParams)
    # Generate the rate series and export rates, making sure they are on the same tz
    utility.generate_rate_series(building_df.index)
    utility.load_export_rates()

    if solarSystemParams is not None:
        # If PV size wasn't provided, calculate it based on the load profile

        # Create a SolarSystem instance using solarSystemParams
        solar_system = SolarSystem(**solarSystemParams)
        solar_system.generate_production_profile(building_df.index)
    else:
        solar_system = None

    # Now, run the overall energy simulation
    if batteryParams is not None:
        energy_system_df = run_sim_w_battery(utility, building_df, solar_system, batteryParams)
    else:
        # Basic math to calculate solar self-consumption and export
        energy_system_df = pd.DataFrame(index=building_df.index)
        energy_system_df['total_load'] = building_df['total_load']
        energy_system_df['solar_production'] = solar_system.pv_prod['Energy Production (kWh)']
        energy_system_df['solar_self_consumption'] = min(energy_system_df['total_load'],
                                                         energy_system_df['solar_production'])
        energy_system_df['solar_export'] = max(energy_system_df['solar_production'] -
                                               energy_system_df['total_load'], 0)
        energy_system_df['electricity_import'] = max(energy_system_df['total_load'] -
                                                     energy_system_df['solar_production'], 0)

        # Other columns for consistency:
        energy_system_df['battery_state'] = 0
        energy_system_df['battery_state_end'] = 0
        energy_system_df['solar_to_battery'] = 0

    financials_dict = calculate_financials(energy_system_df)

    return financials_dict


def run_sim_w_battery(
        utility,
        building_df,
        solar_system,
        battery_params,
):
    """

    :param utility: Utility instance
    :param building_df: pd.DataFrame, index is the time series, 'total_load' is the total electricity load
    :param solar_system: SolarSystem instance if exists, None otherwise
    :param battery_params: dict of battery parameters: {'bess_size': float, 'bess_efficiency': float,
    'bess_max_power': float, 'control_strategy': str ['standard', 'peak_discharge']}
    :return: pd.DataFrame with the following columns:
    'total_load', 'solar_production', 'battery_state', 'battery_state_end', 'solar_self_consumption', 'solar_export',
    """
    energy_system_df = pd.DataFrame(index=building_df.index)
    energy_system_df['total_load'] = building_df['total_load']

    if solar_system is not None:
        energy_system_df['solar_production'] = solar_system.pv_prod['Energy Production (kWh)']
    else:
        energy_system_df['solar_production'] = 0

    energy_system_df['battery_state'] = 0
    energy_system_df['battery_state_end'] = 0
    energy_system_df['solar_self_consumption'] = 0
    energy_system_df['solar_to_battery'] = 0
    energy_system_df['solar_export'] = 0
    energy_system_df['electricity_import'] = 0

    for (i, ts) in enumerate(energy_system_df.index):
        if i == 0:
            energy_system_df['battery_state'].iloc[i] = 0
        else:
            energy_system_df['battery_state'].iloc[i] = energy_system_df['battery_state_end'].iloc[i-1]

        solar_production = energy_system_df.loc[ts, 'solar_production']
        demand = energy_system_df.loc[ts, 'total_load']
        solar_consumed = min(solar_production, demand)
        solar_excess = max(solar_production - demand, 0)
        battery_spare_capacity = battery_params['bess_size'] - energy_system_df.loc[ts, 'battery_state']
        solar_stored = min(solar_excess, battery_spare_capacity, battery_params['bess_max_power'])
        solar_exported = max(solar_excess - solar_stored, 0)

        remaining_demand = max(demand - solar_consumed, 0)

        if battery_params['control_strategy'] == "standard":
            # Standard control strategy
            # Charge the battery with excess solar
            # Discharge the battery to meet demand
            battery_discharge = min(remaining_demand,
                                    energy_system_df.loc[ts, 'battery_state'],
                                    battery_params['bess_max_power'])
        elif battery_params['control_strategy'] == "peak_discharge":
            # Only discharge during peak hour
            if utility.is_peak(ts):
                battery_discharge = min(remaining_demand,
                                        energy_system_df.loc[ts, 'battery_state'],
                                        battery_params['bess_max_power'])
            else:
                battery_discharge = 0

        electricity_import = remaining_demand - battery_discharge

        # New battery state
        battery_state_end = energy_system_df.loc[ts, 'battery_state'] + solar_stored - battery_discharge
        # Check to make sure our math worked:
        if battery_state_end < 0:
            print("Battery state end is negative")
            raise Exception
        elif battery_state_end > battery_params['bess_size']:
            print("Battery state end is greater than battery size")
            raise Exception

        energy_system_df.loc[ts, 'battery_state_end'] = battery_state_end
        energy_system_df.loc[ts, 'solar_self_consumption'] = solar_consumed
        energy_system_df.loc[ts, 'solar_to_battery'] = solar_excess
        energy_system_df.loc[ts, 'solar_export'] = solar_exported
        energy_system_df.loc[ts, 'electricity_import'] = electricity_import

    return energy_system_df


def calculate_financials(energy_system_df):
    """
    Calculate the financials of the energy system
    :param energy_system_df: pd.DataFrame with the following columns:
    'total_load', 'solar_production', 'battery_state', 'battery_state_end', 'solar_self_consumption', 'solar_export',
    'electricity_import'
    :return: dict of financials
    """
    financials = {}
    financials['total_load'] = energy_system_df['total_load'].sum()
    financials['solar_production'] = energy_system_df['solar_production'].sum()
    financials['solar_self_consumption'] = energy_system_df['solar_self_consumption'].sum()
    financials['solar_export'] = energy_system_df['solar_export'].sum()
    financials['electricity_import'] = energy_system_df['electricity_import'].sum()

    return financials


def get_params(
    location='Los Angeles',
    load_profile_house=False,
    load_profile_house_annual_total="",
    load_profile_units=False,
    load_profile_units_num="",
    load_profile_csv = None,
    utility_plan="SCE Time of Use",
    measures_solar=False,
    measures_solar_size="",
    measures_battery=False,
    measures_battery_size="",
):
    """
    Get the parameters for the energy simulation based on basic inputs (e.g. from app)
    :param location: str, location of the building. Currently only 'Los Angeles' is supported
    :param load_profile_house: bool, whether to include the house load profile
    :param load_profile_house_annual_total: optional str to specify total house load. Leave blank to use default.
    :param load_profile_units: bool, whether to include the units load profile
    :param load_profile_units_num: optional str to specify number of units. Default will be 1.
    :param load_profile_csv: optional str, path to a custom load profile CSV.
            If provided, uses this over default house/unit profiles
    :param utility_plan: str, the utility rate plan to use. Currently only 'SCE Time of Use' is supported
    :param measures_solar: bool, whether PV will be a building measure
    :param measures_solar_size: optional str to specify the size of the PV system
    :param measures_battery: bool, whether battery energy storage will be a building measure
    :param measures_battery_size: optional str to specify the size of the battery system
    :param measures_hpwh: bool, whether replacing gas water heater with HPWH will be a building measure
    :return:
    """
    # First check the utility rate plan
    if utility_plan == "SCE Time of Use":
        utilityParams = {
            "util": "SCE",
            "rate_plan": "tou_d_4"
        }
    else:
        raise Exception

    # Then, load the load profile:
    baselineLoadProfileParams = {
        "load_csv_path": load_profile_csv,
        "house_load": load_profile_house,
        "units_load": load_profile_units,
    }
    if load_profile_house_annual_total != "":
        baselineLoadProfileParams["house_load_annual_total"] = float(load_profile_house_annual_total)
    if load_profile_units_num != "":
        baselineLoadProfileParams["units_load_num"] = int(load_profile_units_num)

    if measures_solar:
        solarSystemParams = {
            "location": location,
        }
        if measures_solar_size != "":
            solarSystemParams["pv_size"] = float(measures_solar_size)
        else:
            solarSystemParams["pv_size"] = None  # Will calculate default size later
    else:
        solarSystemParams = None

    if measures_battery:
        batteryParams = {
            "bess_efficiency": 0.9,
            "bess_max_power": 5,
            "control_strategy": "standard"
        }
        if measures_battery_size != "":
            batteryParams["bess_size"] = float(measures_battery_size)
        else:
            batteryParams["bess_size"] = None  # Will calculate default size later
    else:
        batteryParams = None

    params = {
        "utilityParams": utilityParams,
        "baselineLoadProfileParams": baselineLoadProfileParams,
        "solarSystemParams": solarSystemParams,
        "batteryParams": batteryParams
    }
    return params


if __name__ == '__main__':
    # Example usage -- mirroring
    pass
