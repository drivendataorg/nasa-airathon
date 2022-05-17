"""usage: python create_cfg.py <period_name>
<period_name>: name for the new period of data

"""

import os
import sys
import yaml

if __name__ == '__main__':
    
    period_name = sys.argv[1]

    # Creating necessary folders for AOD data
    os.makedirs(f'data/processed/{period_name}', exist_ok=True)
    os.makedirs(f'data/raw/aod/test/{period_name}/maiac', exist_ok=True)
    os.makedirs(f'data/interm/maiac/assembled_csv/test/{period_name}', exist_ok=True)
    os.makedirs(f'data/interm/maiac/extracted_vars/test/{period_name}', exist_ok=True)

    # Creating necessary folders for GFS data
    for loc in ['dl', 'la', 'tp']:
        os.makedirs(f'data/raw/gfs/downloaded_files/{loc}/{period_name}/group_1', exist_ok=True)
        os.makedirs(f'data/raw/gfs/downloaded_files/{loc}/{period_name}/group_2', exist_ok=True)
        os.makedirs(f'data/interm/gfs/merged_csv/{loc}/{period_name}', exist_ok=True)


    # Defining paths
    
    cfg = {'period_name':period_name,
    'path_grid_metadata': 'data/raw/aod/grid_metadata.csv',
    'path_shapefiles': 'data/interm/shape_files',
    'path_maiac_hdf_files': f'data/raw/aod/test/{period_name}/maiac',
    'path_maiac_extracted_vars': f'data/interm/maiac/extracted_vars/test/{period_name}',
    'path_maiac_assembled_csv': f'data/interm/maiac/assembled_csv/test/{period_name}',
    'path_save_final_view': f'data/processed/{period_name}',
    'path_raw_gfs': f'data/raw/gfs/downloaded_files', #{loc}/{period_name}',
    'path_gfs_save_merged': f'data/interm/gfs/merged_csv', # {loc}/{period_name}',
    'aod_csv_filename': f'test_aod_cwv_{period_name}.csv',
    'aod_and_gfs_filename': f'final_aod_and_gfs_{period_name}.csv',
    'path_labels':'data/raw/aod/submission_format.csv',
    'run_in_parallel': True,
    'saved_final_rf_model':f'models/rf_{period_name}_joblib.pkl',
    'saved_final_gbr_model':f'models/grb_{period_name}_joblib.pkl',
    'order_of_columns': 'models/order_of_columns.txt',
    'city_mean_enc_mappings': f'models/saved_encodings/city_mean_enc_mappings_{period_name}_joblib.pkl',
    'global_mean_enc': f'models/saved_encodings/global_mean_enc_{period_name}.txt',
    'le_mappings':f'models/saved_encodings/le_mappings_{period_name}_joblib.pkl',
    'mean_enc_mappings':f'models/saved_encodings/mean_enc_mappings_{period_name}_joblib.pkl'
    }

    with open(f'cfg/{period_name}.yml', 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    




    
