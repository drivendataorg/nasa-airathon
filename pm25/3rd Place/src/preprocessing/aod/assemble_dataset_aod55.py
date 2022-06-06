"""
to run the script:
python assemble_dataset.py <path_extracted_vars> <save_path>
e.g. <path_extracted_vars>: extracted_vars/train/maiac/
<save_path>: assembled_csv

"""
import os
import glob
from pathlib import Path
import sys
import pandas as pd
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool, freeze_support, cpu_count

def extract_var(location: str) -> None:
    """Forms the dataset by extracting each variable from its associated file and stores as csv for the location

    Args:
        location (str): Area of interest / City
    """

    #template_df = pd.DataFrame(columns = ['filename', 'location', 'grid_id', var_of_interest, 'file_n'])

    
    path = Path(f'{home}/{location}') 
    grid_ids = list(path.glob('*'))    

    res = []
    for grid_id in grid_ids:
        
        filename_folders = list(grid_id.glob('*'))
        for filename in filename_folders:
            
            csv_list = [i for i in list(filename.glob('*')) if var_of_interest in i.name]

            file_n = 0 
            for csv_file in csv_list:
                temp_df = pd.read_csv(csv_file)
                
                avg_val = np.nan
                min_v = np.nan
                max_v = np.nan
                percentile_95 = np.nan
                variance = np.nan
                std = np.nan

                try: 
                    avg_val = temp_df[var_of_interest].mean()
                    min_v = temp_df[var_of_interest].min()
                    max_v = temp_df[var_of_interest].max()
                    variance = temp_df[var_of_interest].var()
                    std = temp_df[var_of_interest].std()
                    percentile_95 = temp_df[var_of_interest].quantile(0.95)
                except:
                    pass

                res.append([filename.name, location, grid_id.name, avg_val, min_v, max_v, variance, std, percentile_95, file_n])
                file_n += 1

    df = pd.DataFrame(res, columns = ['filename', 'location', 'grid_id', f'{var_of_interest}_avg', f'{var_of_interest}_min',
    f'{var_of_interest}_max', f'{var_of_interest}_var', f'{var_of_interest}_std', f'{var_of_interest}_p95', 'file_n'])
    os.makedirs(save_path, exist_ok = True)
    df.to_csv(f'{save_path}/{location}_{var_of_interest}_descriptive.csv', index = None)


home = sys.argv[1] # e.g. extracted_vars/train/maiac/
save_path = sys.argv[2] 
var_of_interest = 'Optical_Depth_055'


if __name__ == "__main__":
    
    locations = os.listdir(home)

    freeze_support()
    
    pool = Pool(cpu_count()) 
    for _ in tqdm(pool.imap_unordered(extract_var, locations), total=len(locations)):
        pass
    


