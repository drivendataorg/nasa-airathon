"""usage: python create_cfg.py <period_name>
<period_name>: name for the new period of data

"""

import os
import sys
import yaml
import shutil

if __name__ == '__main__':
    
    period_name = sys.argv[1]

    # Deleting folders for AOD data
    shutil.rmtree(f'data/raw/aod/test/{period_name}')
    shutil.rmtree(f'data/interm/maiac/assembled_csv/test/{period_name}')
    shutil.rmtree(f'data/interm/maiac/extracted_vars/test/{period_name}')

    # Creating necessary folders for GFS data
    for loc in ['dl', 'la', 'tp']:
        shutil.rmtree(f'data/raw/gfs/downloaded_files/{loc}/{period_name}')
        shutil.rmtree(f'data/interm/gfs/merged_csv/{loc}/{period_name}')

    




    
