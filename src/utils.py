import xarray as xr
import pickle
import smogn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def read_datacube(file_path, silent=False, select_north=True):
    if not silent:
        print('file opening...')
    # open datasets
    file1 = xr.open_dataset(file_path, decode_times=False)

    if not silent:
        print(list(file1.data_vars))
        print(list(file1.coords))
    if select_north:
        file1 = file1.where(file1.lon >= 128)
        file1 = file1.where(file1.lon <= 137)
        file1 = file1.where(file1.lat >= -16)
        file1 = file1.where(file1.lat <= -11.5)

    if not silent:
        print('file loading complete')
        print('now converting datasets to dataframe')
    # convert dataset to pandas dataframe
    features = file1.to_dataframe()
    features.reset_index(inplace=True)  # reset index, move lon and lat to columns form indices
    pd.set_option('display.expand_frame_repr', False)
    features = features[features['GFED_regions'] == 14]  # Australia 14

    # filter:
    features = features[features.topo == 1]  # by topography to remove all ocean values
    features = features.replace(-9997, 0)  # na -> 0

    del_list = ['lon', 'lat', 'time', 'Livestock', 'road_density', 'topo', 'Distance_to_populated_areas', 'NLDI']
    for col in del_list:
        del features[col]

    fill_list = features.columns
    for col in fill_list:
        features[col] = features[col].fillna(0)

    # cast ignitions to integer
    #features["ignitions"] = features["ignitions"].astype(int)
    #features["ignitions"] = (features["ignitions"].values > 0).astype(np.uint8)

    return features


def run_smogn(file_path, parallel=True, rel_thresh=0.1, silent=False, features=None):
    if features is None:
        features = read_datacube(file_path, silent=silent)

    # SMOGN data
    if not silent:
        print('Running Smogn...')

    rf_resampled = smogn.smoter(
        data=features.reset_index(drop=True),
        y='ignitions',
        k=5,  ## positive integer (k < n)
        pert=0.05,  ## real number (0 < R < 1)
        samp_method='extreme',  ## string ('balance' or 'extreme')
        drop_na_col=True,  ## boolean (True or False)
        drop_na_row=True,  ## boolean (True or False)
        replace=False,  ## boolean (True or False)
        parallel=parallel,
        silent=silent,

        ## phi relevance arguments
        # 1:0.6,2:0.8,3:0.8
        rel_thres=rel_thresh,
        ## real number (0 < R < 1) It specifies the threshold of rarity. The higher the threshold, the higher the over/under-sampling boundary.
        # rel_method='auto',  ## string ('auto' or 'manual')
        # rel_xtrm_type='high',  ## unused (rel_method = 'manual')
        # rel_coef=2.25,  ## unused (rel_method = 'manual')
        # rel_ctrl_pts_rg=strategy_smogn  ## 2d array (format: [x, y])
    )

    pickle.dump(rf_resampled, open('../out/smogn_resampled.pkl', 'wb'))
    if not silent:
        print('SMOGN complete...')

        print('non zeros in new data:')
        print(rf_resampled['ignitions'].astype(bool).sum(axis=0))
        print('non zeros in old data:')
        print(features['ignitions'].astype(bool).sum(axis=0))
        # print(smogn.box_plot_stats(features_r['ignitions'])['stats'])
        # print(smogn.box_plot_stats(rf_resampled['ignitions'])['stats'])

        # plot y distribution
        sns.kdeplot(features['ignitions'], label="Original")
        sns.kdeplot(rf_resampled['ignitions'], label="Modified")
        plt.legend()
        plt.savefig('../out/Smogn_test.png', bbox_inches='tight', dpi=1080)
        plt.show()
