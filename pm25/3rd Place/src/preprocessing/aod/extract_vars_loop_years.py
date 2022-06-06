import os

years = ['2017', '2018', '2021']
path_hdf_files = 'data/test/maiac'
save_path = 'assembled_csv'
split_type = 'test'
path_shapefiles = 'shape_files'
for year in years:
  os.system("python extract_vars.py {} {} {}".format(f'{path_hdf_files}/{year}', path_shapefiles, split_type))