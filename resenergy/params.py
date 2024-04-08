# Defaults for the base 15-unit sizing for Sanco, no load-shifting
DEFAULT_HW_PARAMS = {
    'incomingT_F': 50.,
    'magnitudeStat': 30.2,
    'gpdpp': 28.7,
    'supplyT_F': 120,
    'storageT_F': 145,
    'percentUseable': 0.85,
    'aquaFract': 0.4,
    'schematic': 'swingtank',
    'buildingType': 'multi_family',
    'zipCode': 90803,
    'nBR': [0, 10, 5],
    'doLoadShift': False,
    'annual': True,

    # Other items we probably won't change for a while
    'Wapt': 100,
    'defrostFactor': 1,
    'compRuntime_hr': 16,
    'loadUpHours': 3,

    # Default sizing
    'systemModel': "MODELS_SANCO2_C_SP",
    'numHeatPumps': 3,
    'PVol_G_atStorageT': 180.95,
    'TMVol_G': 40,
    'TMCap_kW': 2.625,
    'safetyTM': 1.75,

    ## Unknown:
    # ignoreShortCycleEr = False,
    # useHPWHsimPrefMap = False
}
DEFAULT_HW_PARAMS['nApt'] = sum(DEFAULT_HW_PARAMS['nBR'])

DEFAULT_PARALLEL_HW_PARAMS = {
    'incomingT_F': 50.,
    'magnitudeStat': 30.2,
    'gpdpp': 28.7,
    'supplyT_F': 120,
    'storageT_F': 145,
    'percentUseable': 0.85,
    'aquaFract': 0.4,
    'schematic': 'paralleltank',
    'buildingType': 'multi_family',
    'zipCode': 90006,
    'nBR': [0, 10, 5],
    'doLoadShift': False,
    'annual': True,
    'Wapt': 100,
    'defrostFactor': 1,
    'compRuntime_hr': 16,
    'loadUpHours': 3,
    'systemModel': "MODELS_SANCO2_C_SP",
    'numHeatPumps': 2,
    'PVol_G_atStorageT': 180.95,
    'TMVol_G': 21,
    'tmModel': "MODELS_SANCO2_C_SP",
    'tmNumHeatPumps': 1,
    'nApt': sum(DEFAULT_HW_PARAMS['nBR'])
}

DEFAULT_LS_HW_PARAMS = {
    'doLoadShift': True,
    'loadShiftSchedule': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    'loadUpT_F': DEFAULT_HW_PARAMS['storageT_F'] + 10,  # Default
    'aquaFractLoadUp': 0.2,
    'aquaFractShed': 0.8,
    'loadShiftPercent': .95
}


# Just a convenience thing for updating parameters
CONFIGS = {
    'solarParams': [
        {
            'bess_size': 0.,
            'pv_size': 0.
        },
        {
            'bess_size': 0.,
            'pv_size': 3.
        },
        {
            'bess_size': 0.,
            'pv_size': 5.
        },
    ],
    'loadShiftProfile': ['pm_std', 'pm_aggr', 'am_pm_std', 'am_pm_aggr', 'none'],
    'utilityParams': ["tou_d_4", "tou_d_5", "tou_d_prime"],
    'hwParams': {
        'schematic': ['swingtank', 'paralleltank'],
        'numHeatPumps': [3, 4],
        'PVol_G_atStorageT': [150, 180, 250, 300],
        'aquaFractLoadUp': [0.1, 0.15, 0.2, 0.25],
        'aquaFract': [.3, .4, .5],
        'aquaFractShed': [.7, .8, .9]
    }
}
