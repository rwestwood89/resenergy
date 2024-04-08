import pandas as pd
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Data file path
solar_path = os.path.join(current_dir, '..', 'data', 'SolarProfiles')


class SolarSystem:
    def __init__(self,
                 pv_size, # kW
                 location='Los Angeles',
                 ):
        self.location = location
        self.pv_size = pv_size
        self.pv_prod = None

    def generate_production_profile(self, ts):
        # Pass in time series, get solar production dataframe
        # ts needs to be local time (Pacific)

        # Grab and read in correct profile
        prof_name = None
        if self.location == "Los Angeles":
            prof_name = "pv_10kW_losangeles.csv"
            # TODO: generate other profiles

        if prof_name is None:
            raise Exception

        sdf = pd.read_csv(os.path.join(solar_path,prof_name), header=30)
        # Index the solar production profile as hour-of-year
        sdf['HourOfYear'] = sdf['Month'].astype(str) + '-' + sdf['Day'].astype(str) + '-' + sdf['Hour'].astype(str)
        sdf = sdf.set_index('HourOfYear')

        sample_array = ts.month.astype(str)+'-'+ts.day.astype(str)+'-'+ts.hour.astype(str)

        self.pv_prod = sdf['AC System Output (W)'].loc[sample_array]
        self.pv_prod = self.pv_prod.reset_index()
        self.pv_prod['Timestamp'] = ts
        self.pv_prod = self.pv_prod.set_index('Timestamp')

        # Scale up the production based on array size
        scale_factor = self.pv_size / 10
        self.pv_prod['AC System Output (W)'] *= scale_factor

        # Convert power to energy generated in each interval along ts
        dt = ts[1] - ts[0]  # in hours
        self.pv_prod['Energy Production (kWh)'] = self.pv_prod['AC System Output (W)']*(dt.total_seconds()/3600/1000)


