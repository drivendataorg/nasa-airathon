"""usage: python create_cfg.py <period_name>
<period_name>: name for the new period of data

"""
import os
import sys
import yaml
from preprocessing.aod.create_shapefiles import *
from preprocessing.aod.extract_vars import *
from preprocessing.aod.hdf_utils import read_hdf, read_datafield, read_attr_and_check, get_lon_lat
from preprocessing.aod.assemble_dataset import *
from preprocessing.aod.merge_csv_aod_desc_cwv import *
from preprocessing.gfs.gfs_prep_group_1 import *
from preprocessing.gfs.gfs_prep_group_2 import *
from preprocessing.forming_final_view import *


if __name__ == '__main__':
    
    cfg_name = sys.argv[1]

    with open(f'cfg/{cfg_name}') as f:
        # use safe_load instead load
        hmp = yaml.safe_load(f)

    path_grid_metadata = hmp['path_grid_metadata']
    path_shapefiles = hmp['path_shapefiles']
    run_in_par = hmp['run_in_parallel']
    
    # Creating Shapefiles
    print('Creating shapefiles')
    convert_poly(path_grid_metadata, path_shapefiles)

    # Extracting features from maiac hdf files
    print('Extracting features from maiac hdf files')
    locations_key = {'tpe':'Taipei', 'dl':'Delhi', 'la':'Los Angeles (SoCAB)'}
    datafs = ['Optical_Depth_055', 'Column_WV'] #'Optical_Depth_047','AOD_Uncertainty','FineModeFraction','AOD_QA','AOD_MODEL','Injection_Height']
    path_maiac_hdf_files = hmp['path_maiac_hdf_files'] #sys.argv[1]

    years = os.listdir(path_maiac_hdf_files)

    save_path_root_extracted = hmp['path_maiac_extracted_vars'] # extracted_vars/{split_type}/{period_name}

    # EDIT
    os.makedirs(save_path_root_extracted, exist_ok=True)

    for year in years:

        path = f'{path_maiac_hdf_files}/{year}'
        save_path_root = save_path_root_extracted
        ext = extract_fields(locations_key, datafs, path, path_shapefiles, save_path_root)

        filenames = os.listdir(path)
        if run_in_par:
            try:
                freeze_support()
                pool = Pool(cpu_count()) 
                for _ in tqdm(pool.imap_unordered(ext.extract, filenames), total=len(filenames)):
                    pass
            except:
                print("Can't run multiprocessing. Switching to a loop version")
                run_in_par = False
        
        if not run_in_par:
            for filename in tqdm(filenames):
                ext.extract(filename)

    
    # Merging extracted features
    print('Merging extracted features')
    path_extracted = f'{save_path_root_extracted}/maiac' #home
    save_path_assembled = hmp['path_maiac_assembled_csv']
    dataf_aod = 'Optical_Depth_055' #,'Optical_Depth_047','AOD_Uncertainty','FineModeFraction','Column_WV','AOD_QA','AOD_MODEL','Injection_Height']
    dataf_cwv = 'Column_WV'

    # EDIT
    os.makedirs(save_path_assembled, exist_ok=True)

    mf_aod = merge_features(path_extracted, save_path_assembled, dataf_aod)
    mf_cwv = merge_features(path_extracted, save_path_assembled, dataf_cwv)

    locations = os.listdir(path_extracted)

    if run_in_par:
            try:
                freeze_support()
                pool = Pool(cpu_count()) 
                for _ in tqdm(pool.imap_unordered(mf_cwv.extract_var, locations), total=len(locations)):
                    pass
            except:
                print("Can't run multiprocessing. Switching to a loop version")
                run_in_par = False

            try:
                freeze_support()
                pool = Pool(cpu_count()) 
                for _ in tqdm(pool.imap_unordered(mf_aod.extract_var_aod_55, locations), total=len(locations)):
                    pass
            except:
                print("Can't run multiprocessing. Switching to a loop version")
                run_in_par = False
        
    if not run_in_par:
        for location in tqdm(locations):
            mf_cwv.extract_var(location)
            mf_aod.extract_var_aod_55(location)


    # Joining train/test_labels with aod and cwv
    print('Joining train/test_labels with aod and cwv')
    path_to_the_assembled_csv = save_path_assembled
    path_train_label = hmp['path_labels']
    save_path = hmp['path_save_final_view']
    save_filename = hmp['aod_csv_filename']

    # EDIT
    os.makedirs(save_path, exist_ok=True)

    join_labels_aod_cwv(path_to_the_assembled_csv, path_train_label, save_path, save_filename)


    # Preparing GFS group 1 data
    print('Preparing GFS data')
    period_name = hmp['period_name']
    root = hmp['path_raw_gfs']
    save_path_root = hmp['path_gfs_save_merged']

    locations = ['la', 'tp', 'dl']
    loc_keys = {'la':'Los Angeles (SoCAB)', 'tp':'Taipei', 'dl':'Delhi'}
    grid_meta = pd.read_csv(path_grid_metadata)
    var_group = 'group_1'
    
    for loc in locations:
        read_path = f'{root}/{loc}/{period_name}/{var_group}'
        save_path = f'{save_path_root}/{loc}/{period_name}'
        # EDIT
        os.makedirs(save_path, exist_ok=True)

        prep_group_1(root, read_path, save_path, loc, loc_keys, grid_meta, var_group)

    var_group = 'group_2'
    
    for loc in locations:
        read_path = f'{root}/{loc}/{period_name}/{var_group}'
        save_path = f'{save_path_root}/{loc}/{period_name}'
        # EDIT
        os.makedirs(save_path, exist_ok=True)

        prep_group_2(root, read_path, save_path, loc, loc_keys, grid_meta, var_group)
        

    # Forming the final view
    print('Forming the final view')
    read_path = hmp['path_gfs_save_merged']
    save_path = hmp['path_save_final_view']
    # EDIT
    os.makedirs(save_path, exist_ok=True)

    aod_desc_path = f'{save_path}/{hmp["aod_csv_filename"]}'
    final_view_save_name = hmp['aod_and_gfs_filename']
    locations = ['la', 'tp', 'dl']    

    form_final_view(aod_desc_path, read_path, save_path, final_view_save_name, locations, period_name)


    




    
