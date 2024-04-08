import numpy as np
import pandas as pd
from pytz import timezone
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.widgets import CheckButtons
from datetime import timedelta
import random

from ecoengine import EcosizerEngine, getListOfModels, SimulationRun, getAnnualSimLSComparison, PrefMapTracker

from resenergy.params import *

W_TO_BTUHR = 3.412142


class HeatpumpWaterHeater:
    def __init__(self,
                 hwParams=None,
                 loadShiftProfile=None,
                 hwLoadProfile=None,
                 ):
        if hwParams is None:
            hwParams = dict(DEFAULT_HW_PARAMS)

        # Required parameter calculations
        hwParams['nApt'] = sum(hwParams['nBR'])  # Make sure this matches
        if hwParams['systemModel'] == 'MODELS_SANCO2_C_SP' and hwParams['numHeatPumps'] is not None:
            # If we specify number of Sanco HPs, need to manually set the output heating capacity
            hwParams['PCap_kW'] = hwParams['numHeatPumps']*3.28

        # do we need to set TMCap_kW?
        if hwParams['schematic'] == 'paralleltank' and \
                hwParams['tmModel'] == 'MODELS_SANCO2_C_SP' and \
                hwParams['tmNumHeatPumps'] is not None:
            hwParams['TMCap_kW'] = hwParams['tmNumHeatPumps']*3.28

        self.schematic = hwParams['schematic']
        self.hwParams = hwParams
        self.loadShiftProfileStr = loadShiftProfile  # Save (string) name for profile
        if loadShiftProfile is not None:
            self.setLoadShiftSchedule(loadShiftProfile)
        self.hwLoadProfile = hwLoadProfile

        self.runComplete = False
        self.annualCOP = None

        ##
        ## Internal objects / references
        # HPWH simulation
        self.hpwhEngine = None
        self.hwSimRun = None
        self.hpwh_df = None
        self.building_df = None  # basically hpwh_df with house_baseload columns
        self.building_avg_df = None

        # Utility
        self.localUtil = None

        # Solar
        self.solarSys = None
        self.solar_df = None
        self.solar_avg_df = None

        # Overall outputs
        self.fin_df = None
        self.fin_avg_df = None

        self.res_dict = None

    def setLoadShiftSchedule(self, loadShiftStr):
        if loadShiftStr == 'pm_std':
            # 4-9pm shift
            self.hwParams['loadShiftSchedule'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
                                                  1]
        elif loadShiftStr == 'pm_aggr':
            # 4-11pm shift
            self.hwParams['loadShiftSchedule'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                                  1]
        elif loadShiftStr == 'am_pm_std':
            # 7-10am, 4-9pm shift
            self.hwParams['loadShiftSchedule'] = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
                                                  1]
        elif loadShiftStr == 'am_pm_aggr':
            # 6-11am, 4-11pm shift
            self.hwParams['loadShiftSchedule'] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]

        elif loadShiftStr == 'none':
            self.hwParams['doLoadShift'] = False

        else:
            raise Exception

    def setRandomHwLoadProfile(self):
        # TODO: have input parameters and generate profile
        self.hwLoadProfile = None

    def runHotWaterSim(self):
        self.hpwhEngine = EcosizerEngine(
            **self.hwParams
        )
        self.hwSimRun = self.hpwhEngine.getSimRun(minuteIntervals=15, nDays=365, exceptOnWaterShortage=False)
        # TODO: turn into catch-exception to return the scenario
        if len(self.hwSimRun.getHWDemand()) != len(self.hwSimRun.getPrimaryVolume()):
            raise Exception('Run Failed')

        if self.schematic=='swingtank':
            data_dict = {
                'HW Demand': self.hwSimRun.getHWDemand(),
                'HW Generation': self.hwSimRun.getHWGeneration(),
                'HW Out (swing)': self.hwSimRun.gethwOutSwing(),
                'TM Temperature': self.hwSimRun.getTMTemp(),

                'Cap In': self.hwSimRun.getCapIn(),
                'Cap Out': self.hwSimRun.getCapOut(),

                'Primary Generation': self.hwSimRun.getPrimaryGeneration(),
                'Primary Run': self.hwSimRun.getPrimaryRun(),
                'Primary Volume': self.hwSimRun.getPrimaryVolume(),

                'TM Cap In': self.hwSimRun.getTMCapIn(),
                'TM Cap Out': self.hwSimRun.getTMCapOut(),
                'TM Run': self.hwSimRun.getTMRun(),

            }

        elif self.schematic=='paralleltank':
            data_dict = {
                'HW Demand': self.hwSimRun.getHWDemand(),
                'HW Generation': self.hwSimRun.getHWGeneration(),
                # to confirm: this includes the parallel tank

                'TM Temperature': self.hwSimRun.getTMTemp(),

                'Cap In': self.hwSimRun.getCapIn(),
                'Cap Out': self.hwSimRun.getCapOut(),

                'Primary Generation': self.hwSimRun.getPrimaryGeneration(),
                'Primary Run': self.hwSimRun.getPrimaryRun(),
                'Primary Volume': self.hwSimRun.getPrimaryVolume(),

                'TM Cap In': self.hwSimRun.getTMCapIn(),
                'TM Cap Out': self.hwSimRun.getTMCapOut(),
                'TM Run': self.hwSimRun.getTMRun(),

            }

        datetime_index = pd.date_range(start='2018-01-01', periods=len(data_dict['HW Demand']), freq='15T',
                                       tz='US/Pacific')
        df = pd.DataFrame(data_dict, index=datetime_index)

        # kW * minutes running / (60 min/hr) ==> kWh
        df['Energy - Primary (kWh)'] = (df['Primary Run'] * df['Cap In']) / 60.
        df['Energy - TM (kWh)'] = (df['TM Run'] * df['TM Cap In']) / 60.
        df['Energy Water Heater (kWh)'] = df['Energy - Primary (kWh)'] + df['Energy - TM (kWh)']

        self.hpwh_df = df


    def simSolarAndWaterHeating(self):
        ## Get house load
        #  - For now, use NREL un-electrified base load and scale
        house_load_profile = load_profile.load_resstock_profile(
            load_profile.Profiles.HotDryBaseHouse,
            total_annual=self.houseAnnualBaseload
        )
        # For convenience, wrap around so the whole datetimeindex is in 2018.
        index2017 = house_load_profile[house_load_profile.index.year == 2017]  # get any 2017 data
        index2017.index = index2017.index + pd.DateOffset(years=1)  # roll over to 2018
        house_load_profile = pd.concat([house_load_profile, index2017])  # concatenate with prior data
        house_load_profile = house_load_profile[house_load_profile.index.year == 2018]  # keep only 2017

        house_load_profile = house_load_profile.rename(columns={'total': 'House Baseload'})

        ## Hot water sim
        self.runHotWaterSim()
        # Now, merge the two onto the same time index
        building_df = pd.merge(house_load_profile, self.hpwh_df, how='left', left_index=True, right_index=True)

        # Sum the electricity loads
        building_df['total'] = building_df['House Baseload'] + building_df['Energy Water Heater (kWh)']

        ## Utility Rate
        if self.utilityParams is None:
            self.utilityParams = {
                'util': "SCE",
                'plan': "tou_d_prime"
            }
        self.localUtil = utility.Utility(**self.utilityParams)
        # Pass your load profile time series to sample the rate plan
        self.localUtil.generate_rate_series(house_load_profile.index)
        # Load in export rates
        self.localUtil.load_export_rates()

        ## Solar
        if self.solarParams is None:
            bess_size = 0  # Start with no storage
            # Quick and dirty sizing
            pv_size = 3.2 * float(self.hwParams['nApt']) / 15.  # hot water heating load
            # Add 1kW solar for every 1,600 kWh energy annually for house electrical baseload
            pv_size += building_df['House Baseload'].sum() / 1600.
            self.solarParams = {
                'bess_size': bess_size,
                'pv_size': pv_size
            }
        self.solarSys = solar_prod.SolarSystem(**self.solarParams)
        self.solarSys.generate_production_profile(building_df.index)

        ## Energy sim
        res_dict, solar_df, fin_df_bk = run_scenario(
            util_inst=self.localUtil,
            elec_load_profile=building_df,  # just needs 'total' column
            solar_sys_inst=self.solarSys
        )

        # Don't want most of the other fin_df (made for residential). Redo for WH.
        fin_df = pd.DataFrame(index=solar_df.index)
        fin_df['Import Rate'] = fin_df_bk['Import Rate']
        fin_df['Export Rate'] = fin_df_bk['Export Rate']
        fin_df['Peak Hour'] = fin_df_bk['Peak Hour']

        fin_df['WH Total'] = building_df['Energy Water Heater (kWh)']
        fin_df['House Baseload'] = building_df['House Baseload']

        fin_df['Solar Production'] = solar_df['Solar Production']
        fin_df['Solar Consumed On-Site'] = solar_df['Solar Consumed On-site']
        fin_df['Solar to WH'] = fin_df[['WH Total', 'Solar Consumed On-Site']].min(axis=1)
        fin_df['Import to WH'] = fin_df['WH Total'] - fin_df['Solar to WH']
        fin_df['Import to WH Peak'] = fin_df['Import to WH'] * fin_df_bk['Peak Hour']
        fin_df['Import to WH Off-Peak'] = fin_df['Import to WH'] - fin_df['Import to WH Peak']
        fin_df['WH Import Cost'] = fin_df['Import to WH'] * fin_df['Import Rate']

        fin_df['Solar to Baseload'] = solar_df["Solar Consumed On-site"] - fin_df['Solar to WH']
        fin_df['Solar to Baseload Value'] = fin_df['Solar to Baseload'] * fin_df['Import Rate']

        fin_df['Solar Exported'] = solar_df["Solar Exported"]
        fin_df['Solar Exported Value'] = fin_df['Solar Exported'] * fin_df['Export Rate']

        print("---WATER HEATER---")
        self.annualCOP = self.hwSimRun.getAnnualCOP()
        print("WH Annual COP: %0.012f" % (self.annualCOP))
        print("Total HPWH energy usage = %d" % fin_df['WH Total'].sum())

        perc_wh_from_solar = fin_df['Solar to WH'].sum() / fin_df['WH Total'].sum()
        print("WH from Solar: %0.01f%%" % (perc_wh_from_solar * 100.))
        res_dict['perc_wh_from_solar'] = perc_wh_from_solar

        perc_wh_from_peak_import = fin_df['Import to WH Peak'].sum() / fin_df['WH Total'].sum()
        print("WH from Peak Import: %0.01f%%" % (perc_wh_from_peak_import * 100.))
        res_dict['perc_wh_from_peak_import'] = perc_wh_from_peak_import

        perc_wh_from_offpeak_import = fin_df['Import to WH Off-Peak'].sum() / fin_df['WH Total'].sum()
        print("WH from Off-Peak Import: %0.01f%%" % (perc_wh_from_offpeak_import * 100.))
        res_dict['perc_wh_from_offpeak_import'] = perc_wh_from_offpeak_import

        if fin_df['Solar Production'].sum()>0:
            print("---SOLAR END USE---")
            perc_solar_to_wh = fin_df['Solar to WH'].sum() / fin_df['Solar Production'].sum()
            print("Solar to WH: %0.01f%%" % (perc_solar_to_wh * 100.))
            res_dict['perc_solar_to_wh'] = perc_solar_to_wh

            perc_solar_to_baseload = fin_df['Solar to Baseload'].sum() / fin_df['Solar Production'].sum()
            print("Solar to baseload: %0.01f%%" % (perc_solar_to_baseload * 100.))
            res_dict['perc_solar_to_baseload'] = perc_solar_to_baseload

            perc_solar_exported = fin_df['Solar Exported'].sum() / fin_df['Solar Production'].sum()
            print("Solar exported: %0.01f%%" % (perc_solar_exported * 100.))
            res_dict['perc_solar_exported'] = perc_solar_exported
        else:
            res_dict['perc_solar_to_wh'] = 0
            res_dict['perc_solar_to_baseload'] = 0
            res_dict['perc_solar_exported'] = 0

        print(res_dict)

        self.solar_df = solar_df
        self.building_df = building_df
        self.fin_df = fin_df
        self.res_dict = res_dict

        self.get_df_avg_day()

    def estimateFinancials(
        self,
        gas_rate_per_kwh=0.07,  # California average
        gas_eff = 0.9,
        solar_cost_per_watt=2.85,
        battery_storage_per_kwh=1215,
        apr=0.07,
        years=9
    ):
        financial_dict = {}

        solar_cost = self.solarParams['pv_size'] * 1000 * solar_cost_per_watt
        bess_cost = self.solarParams['bess_size'] * battery_storage_per_kwh
        financial_dict["Solar + Battery CapEx"] = solar_cost + bess_cost
        financial_dict["Annual solar payment"] = apr * financial_dict["Solar + Battery CapEx"] * (1. / (1 - (1 + apr) ** (-years)))

        financial_dict["Cost of Peak Import"] = (self.fin_df['Import to WH Peak']*self.fin_df['Import Rate']).sum()
        financial_dict["Cost of Off-Peak Import"] = (self.fin_df['Import to WH Off-Peak']*self.fin_df['Import Rate']).sum()
        financial_dict["Total Cost of WH Import"] = financial_dict["Cost of Peak Import"] + financial_dict["Cost of Off-Peak Import"]

        financial_dict["Value of Solar Export"] = self.fin_df['Solar Exported Value'].sum()
        financial_dict["Value of Baseload Offset"] = self.fin_df['Solar to Baseload Value'].sum()

        financial_dict["Est Gas Usage"] = (self.annualCOP / gas_eff)*self.fin_df['WH Total'].sum()
        financial_dict["Est Gas Cost"] = financial_dict["Est Gas Usage"] * gas_rate_per_kwh

        financial_dict["Net Cost"] = financial_dict["Total Cost of WH Import"] - \
                                     financial_dict["Value of Solar Export"] - \
                                     financial_dict["Value of Baseload Offset"] - \
                                     financial_dict["Est Gas Cost"]

        financial_dict["Net of solar financing"] = financial_dict["Net Cost"] + financial_dict["Annual solar payment"]

        for key, value in financial_dict.items():
            print(f"{key}: {value}")

        return financial_dict

    def plot_annual_net_energy(self):
        daily_sum_df = self.fin_df.resample('D').sum()
        # Plot total solar export per day and total grid import for WH per day

        f, ax = plt.subplots()
        daily_sum_df.plot(y=['Solar Exported', 'Import to WH'], ax=ax)
        plt.show()

    def plot_day_solar_to_wh(self, day):
        tzinfo = self.fin_df.index.tzinfo
        start_day = tzinfo.localize(pd.to_datetime(day))
        end_day = start_day + timedelta(days=1)

        f, ax = plt.subplots()
        twin = ax.twinx()

        self.fin_df.loc[start_day:end_day].plot(y=["House Baseload", "WH Total", "Solar Production",
                                                   "Solar Consumed On-Site", "Solar Exported"], ax=ax)
        ax.legend(loc='upper right')
        self.building_df.loc[start_day:end_day].plot(y="Primary Volume", color='darkmagenta', linestyle='--', linewidth=0.8,
                                                ax=twin)
        twin.legend(loc='lower right')
        plt.show()



def runHotWaterSim(hw_params):
    # Create
    hpwh = EcosizerEngine(
        **hw_params
    )
    simRun = hpwh.getSimRun(minuteIntervals=15, nDays=365, exceptOnWaterShortage=False)
    # TODO: turn into catch-exception to return the scenario

    data_dict = {
        'HW Demand': simRun.getHWDemand(),
        'HW Generation': simRun.getHWGeneration(),
        'HW Out (swing)': simRun.gethwOutSwing(),
        'TM Temperature': simRun.getTMTemp(),

        'Cap In': simRun.getCapIn(),
        'Cap Out': simRun.getCapOut(),

        'Primary Generation': simRun.getPrimaryGeneration(),
        'Primary Run': simRun.getPrimaryRun(),
        'Primary Volume': simRun.getPrimaryVolume(),

        'TM Cap In': simRun.getTMCapIn(),
        'TM Cap Out': simRun.getTMCapOut(),
        'TM Run': simRun.getTMRun(),

    }
    datetime_index = pd.date_range(start='2018-01-01', periods=len(data_dict['HW Demand']), freq='15T', tz='US/Pacific')
    df = pd.DataFrame(data_dict, index=datetime_index)

    # kW * minutes running / (60 min/hr) ==> kWh
    df['Energy - Primary (kWh)'] = (df['Primary Run'] * df['Cap In']) / 60.
    df['Energy - TM (kWh)'] = (df['TM Run'] * df['TM Cap In']) / 60.
    df['Energy Water Heater (kWh)'] = df['Energy - Primary (kWh)'] + df['Energy - TM (kWh)']

    return df, simRun


def simScenario(hw_params=None, hw_load_prof_gen=True, annual_baseload=0, solar_params=None):
    if hw_params is None:
        hw_params = DEFAULT_HW_PARAMS

    if hw_load_prof_gen:
        # Get a hot water load profile
        hw_demand_profile = getRandomHWLoadProfile()
        # Add to the hw_params
        hw_params['loadShape'] = hw_demand_profile
        # hw_params['avgLoadShape']
        # may also want to scale hw_params['gpdpp']

    # Get house load
    #  - For now, use NREL un-electrified base load and scale
    house_load_profile = load_profile.load_resstock_profile(
        load_profile.Profiles.HotDryBaseHouse,
        total_annual=annual_baseload
    )
    # For convenience, wrap around so the whole datetimeindex is in 2018.
    index2017 = house_load_profile[house_load_profile.index.year == 2017]  # get any 2017 data
    index2017.index = index2017.index + pd.DateOffset(years=1)  # roll over to 2018
    house_load_profile = pd.concat([house_load_profile, index2017])  # concatenate with prior data
    house_load_profile = house_load_profile[house_load_profile.index.year == 2018]  # keep only 2017

    house_load_profile = house_load_profile.rename(columns={'total': 'House Baseload'})

    # Run hot water sim
    hw_df, simRun = runHotWaterSim(hw_params)

    # Now, merge the two onto the same time index
    building_df = pd.merge(house_load_profile, hw_df, how='left', left_index=True, right_index=True)

    # Sum the electricity loads
    building_df['total'] = building_df['House Baseload'] + building_df['Energy Water Heater (kWh)']

    util = "SCE"
    plan = "tou_d_prime"
    local_util = utility.Utility(
        util=util,
        plan=plan
    )
    # Pass your load profile time series to sample the rate plan
    local_util.generate_rate_series(house_load_profile.index)
    # Load in export rates
    local_util.load_export_rates()

    # TODO: parameterize
    if solar_params is None:
        bess_size = 0  # Start with no storage
        # Quick and dirty sizing
        pv_size = 3.2 * float(hw_params['nApt']) / 15.  # hot water heating load
        # Add 1kW solar for every 1,600 kWh energy annually for house electrical baseload
        pv_size += building_df['House Baseload'].sum() / 1600.
    else:
        bess_size = solar_params['bess_size']
        pv_size = solar_params['pv_size']

    print("PV size: %0.01f" % pv_size)
    solar_sys = solar_prod.SolarSystem(
        pv_size=pv_size,
        bess_size=bess_size
    )
    solar_sys.generate_production_profile(building_df.index)

    # Run sim
    res_dict, solar_df, fin_df_bk = run_scenario(
        util_inst=local_util,
        elec_load_profile=building_df,  # just needs 'total' column
        solar_sys_inst=solar_sys
    )

    # Don't want most of the other fin_df (made for residential). Redo for WH.
    fin_df = pd.DataFrame(index=solar_df.index)
    fin_df['Import Rate'] = fin_df_bk['Import Rate']
    fin_df['Export Rate'] = fin_df_bk['Export Rate']
    fin_df['Peak Hour'] = fin_df_bk['Peak Hour']

    fin_df['WH Total'] = building_df['Energy Water Heater (kWh)']
    fin_df['House Baseload'] = building_df['House Baseload']

    fin_df['Solar Production'] = solar_df['Solar Production']
    fin_df['Solar Consumed On-Site'] = solar_df['Solar Consumed On-site']
    fin_df['Solar to WH'] = fin_df[['WH Total', 'Solar Consumed On-Site']].min(axis=1)
    fin_df['Import to WH'] = fin_df['WH Total'] - fin_df['Solar to WH']
    fin_df['Import to WH Peak'] = fin_df['Import to WH'] * fin_df_bk['Peak Hour']
    fin_df['Import to WH Off-Peak'] = fin_df['Import to WH'] - fin_df['Import to WH Peak']
    fin_df['WH Import Cost'] = fin_df['Import to WH'] * fin_df['Import Rate']

    fin_df['Solar to Baseload'] = solar_df["Solar Consumed On-site"] - fin_df['Solar to WH']
    fin_df['Solar to Baseload Value'] = fin_df['Solar to Baseload'] * fin_df['Import Rate']

    fin_df['Solar Exported'] = solar_df["Solar Exported"]
    fin_df['Solar Exported Value'] = fin_df['Solar Exported'] * fin_df['Export Rate']

    print("---WATER HEATER---")
    annual_COP = simRun.getAnnualCOP()
    print("WH Annual COP: %0.012f" % (annual_COP))
    print("Total HPWH energy usage = %d" % fin_df['WH Total'].sum())

    perc_wh_from_solar = fin_df['Solar to WH'].sum() / fin_df['WH Total'].sum()
    print("WH from Solar: %0.01f%%" % (perc_wh_from_solar * 100.))
    perc_wh_from_peak_import = fin_df['Import to WH Peak'].sum() / fin_df['WH Total'].sum()
    print("WH from Peak Import: %0.01f%%" % (perc_wh_from_peak_import * 100.))
    perc_wh_from_offpeak_import = fin_df['Import to WH Off-Peak'].sum() / fin_df['WH Total'].sum()
    print("WH from Off-Peak Import: %0.01f%%" % (perc_wh_from_offpeak_import * 100.))

    print("---SOLAR END USE---")
    perc_solar_to_wh = fin_df['Solar to WH'].sum() / fin_df['Solar Production'].sum()
    print("Solar to WH: %0.01f%%" % (perc_solar_to_wh * 100.))
    perc_solar_to_baseload = fin_df['Solar to Baseload'].sum() / fin_df['Solar Production'].sum()
    print("Solar to baseload: %0.01f%%" % (perc_solar_to_baseload * 100.))
    perc_solar_exported = fin_df['Solar Exported'].sum() / fin_df['Solar Production'].sum()
    print("Solar exported: %0.01f%%" % (perc_solar_exported * 100.))

    print(res_dict)
    fin_df.to_csv('SolarHPWH.csv')

    return res_dict, solar_df, fin_df, building_df, annual_COP

