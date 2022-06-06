"""
to run the script:
python create_shapefiles.py <path_to_the_grid_metadata_csv> <save_path>
"""

import sys
import os
import shapely.wkt
import pandas as pd
import geopandas as gpd


def convert_poly(csv_path: str, save_path_root: str) -> None:
    """Converts polygons from the grid_meatadata.csv to shapefiles

    Expected columns: grid_id, location, timezone, wkt in the csv

    Args:
        csv_path (str): path to the grid_metatdata.csv
    """

    df = pd.read_csv(csv_path)
    home_path = save_path_root
    for i in range(df.shape[0]):
        grid_id, lo, _, wkt = df.iloc[i]

        save_path = f'{home_path}/{lo}/{grid_id}'
        os.makedirs(save_path, exist_ok = True)
        
        # Load polygon and save
        gpd.GeoDataFrame(geometry=[shapely.wkt.loads(wkt)]).to_file(f'{save_path}/{grid_id}.shp')



if __name__ == "__main__":
    convert_poly(sys.argv[1], sys.argv[2])