import os
import datetime
import pandas as pd


sce_plans = ("tou_d_4", "tou_d_5", "tou_d_prime")
# export rates from here: https://osesmo.shinyapps.io/NBT_ECR_Data_Viewer/

data_fname = 'Residential General Market SCE Net Billing Tariff Export Compensation Rates.csv'
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Data file path
sce_export_rate_path = os.path.join(current_dir, '..', 'data', data_fname)


class Utility:
    def __init__(self, util="SCE",
                 rate_plan=None,
                 ):
        self.util = util
        if rate_plan is not None:
            self.plan = rate_plan
        elif self.util == "SCE":
            self.plan = sce_plans[0]
        self.rate_df = None

    def generate_rate_series(self, ts):
        # Pass in time series, generate rate dataframe
        # ts needs to be local time (Pacific)
        self.rate_df = pd.DataFrame()
        self.rate_df['Timestamp'] = ts

        self.get_rate()
        self.rate_df = self.rate_df.set_index('Timestamp')

    def get_rate(self):
        # dt must be a datetime
        # Note: does not include monthly credits / daily charges
        rate = None

        dt = self.rate_df['Timestamp']
        if self.util == "SCE":
            if self.plan == sce_plans[0]:
                print("NOTE: Assuming below baseline credit for now")
                pk_start = datetime.time(hour=16)
                pk_end = datetime.time(hour=21)
                day_start = datetime.time(hour=8)

                rate_dict = {
                    "Summer": {
                        "Weekday": {
                            "Peak": 0.47,
                            "Non-Peak": 0.26
                        },
                        "Weekend": {
                            "Peak": 0.36,
                            "Non-Peak": 0.26
                        }
                    },
                    "Winter": {
                        "Weekday": {
                            "Peak": 0.36,
                            "Pre-Peak": 0.25,
                            "Post-Peak": 0.28,
                        },
                        "Weekend": {
                            "Peak": 0.36,
                            "Pre-Peak": 0.25,
                            "Post-Peak": 0.26,
                        }
                    }
                }

            elif self.plan == sce_plans[1]:
                pk_start = datetime.time(hour=17)
                pk_end = datetime.time(hour=20)
                day_start = datetime.time(hour=8)

                rate_dict = {
                    "Summer": {
                        "Weekday": {
                            "Peak": 0.61,
                            "Non-Peak": 0.26
                        },
                        "Weekend": {
                            "Peak": 0.43,
                            "Non-Peak": 0.26
                        }
                    },
                    "Winter": {
                        "Weekday": {
                            "Peak": 0.49,
                            "Pre-Peak": 0.24,
                            "Post-Peak": 0.29,
                        },
                        "Weekend": {
                            "Peak": 0.49,
                            "Pre-Peak": 0.24,
                            "Post-Peak": 0.29,
                        }
                    }
                }

            elif self.plan == sce_plans[2]:
                pk_start = datetime.time(hour=16)
                pk_end = datetime.time(hour=21)
                day_start = datetime.time(hour=8)


                rate_dict = {
                    "Summer": {
                        "Weekday": {
                            "Peak": 0.62,
                            "Non-Peak": 0.24
                        },
                        "Weekend": {
                            "Peak": 0.37,
                            "Non-Peak": 0.24
                        }
                    },
                    "Winter": {
                        "Weekday": {
                            "Peak": 0.57,
                            "Pre-Peak": 0.22,
                            "Post-Peak": 0.22,
                        },
                        "Weekend": {
                            "Peak": 0.57,
                            "Pre-Peak": 0.22,
                            "Post-Peak": 0.22,
                        }
                    }
                }

            ii_summer = dt.apply(lambda x: x.month in range(6, 10))
            ii_weekday = dt.apply(lambda x: x.weekday() < 5)
            ii_peak = dt.apply(lambda x: pk_start < x.time() < pk_end)
            ii_pre_peak = dt.apply(lambda x: day_start < x.time() < pk_start)
            ii_post_peak = ~(ii_peak | ii_pre_peak)

            # Save these values for easy use in battery management strategy
            self.rate_df['Summer'] = ii_summer
            self.rate_df['Weekday'] = ii_weekday
            self.rate_df['Peak Hour'] = ii_peak
            self.rate_df['Pre-Peak Hour'] = ii_pre_peak
            self.rate_df['Post-Peak Hour'] = ii_post_peak

            # Summer
            #  Weekday
            #   Peak
            rate = ii_summer * ii_weekday * ii_peak * rate_dict["Summer"]["Weekday"]["Peak"]
            #   Non-Peak
            rate += ii_summer * ii_weekday * (~ii_peak) * rate_dict["Summer"]["Weekday"]["Non-Peak"]
            #  Weekend
            #   Peak
            rate += ii_summer * (~ii_weekday) * ii_peak * rate_dict["Summer"]["Weekend"]["Peak"]
            #   Non-Peak
            rate += ii_summer * (~ii_weekday) * (~ii_peak) * rate_dict["Summer"]["Weekend"]["Non-Peak"]

            # Winter
            #  Weekday
            #   Peak
            rate += (~ii_summer) * ii_weekday * ii_peak * rate_dict["Winter"]["Weekday"]["Peak"]
            #   Pre-Peak
            rate += (~ii_summer) * ii_weekday * ii_pre_peak * rate_dict["Winter"]["Weekday"]["Pre-Peak"]
            #   Post-Peak
            rate += (~ii_summer) * ii_weekday * ii_post_peak * rate_dict["Winter"]["Weekday"]["Post-Peak"]
            #   Post-Peak
            #  Weekend
            #   Peak
            rate += (~ii_summer) * (~ii_weekday) * ii_peak * rate_dict["Winter"]["Weekend"]["Peak"]
            #   Pre-Peak
            rate += (~ii_summer) * (~ii_weekday) * ii_pre_peak * rate_dict["Winter"]["Weekend"]["Pre-Peak"]
            #   Post-Peak
            rate += (~ii_summer) * (~ii_weekday) * ii_post_peak * rate_dict["Winter"]["Weekend"]["Post-Peak"]

        if rate is None:
            raise Exception

        self.rate_df['Rate'] = rate

    def load_export_rates(self, year=2023):
        if self.util == "SCE":
            export_df = pd.read_csv(sce_export_rate_path)
        else:
            raise NotImplementedError

        #
        # Special look-up format
        # Month
        export_df["datetime"] = export_df["DateStart"].astype(str) + " " + export_df["TimeStart"].astype(str)
        export_df["datetime"] = pd.to_datetime(export_df["datetime"])
        export_df["weekday"] = export_df["DayTypeStart"].apply(lambda x: 1 if x==1 else 0)

        # Filter out everything except year of interest
        export_df = export_df[export_df["datetime"].dt.year == year]

        export_df["ExportRateIndex"] = export_df["datetime"].dt.month.astype(str) + '-' + \
            export_df["datetime"].dt.hour.astype(str) + '-' + \
            export_df["weekday"].astype(str)
        export_df = export_df.set_index("ExportRateIndex")

        # Ge
        self.rate_df = self.rate_df.reset_index()
        ii_weekday = self.rate_df["Timestamp"].apply(lambda x: 1 if x.weekday() < 5 else 0)
        sample_array = self.rate_df["Timestamp"].dt.month.astype(str) + '-' + \
                       self.rate_df["Timestamp"].dt.hour.astype(str) + '-' + \
                       ii_weekday.astype(str)

        value_df = export_df['Value'].loc[sample_array]
        value_df = value_df.reset_index()
        self.rate_df["Export Rate Value"] = value_df['Value']
        self.rate_df = self.rate_df.set_index('Timestamp')

