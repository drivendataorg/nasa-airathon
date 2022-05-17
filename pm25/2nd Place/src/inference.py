import pandas as pd, numpy as np
import os,sys,random,argparse
from pathlib import Path
from sklearn.model_selection import KFold,TimeSeriesSplit
import sklearn.metrics as skm
import lightgbm as lgb


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Path to root data directory", required=True)
parser.add_argument("--model_dir", help="Directory to save models", required=True, default='models')
parser.add_argument("--stage", help="Stage of data to process, i.e test,prod", required=True) 
parser.add_argument("--subformat_path", help="Path to submission format csv file", required=True) 
parser.add_argument("--output_path", help="Path to output file", default='Submission.csv') 

args = parser.parse_args()
DATA_DIR = args.data_dir 
MODEL_DIR = args.model_dir
STAGE = args.stage 
SUB_FORMAT = args.subformat_path 
OUTPUT_PATH = args.output_path 

DATA_DIR = Path(DATA_DIR) 
MODEL_DIR = Path(MODEL_DIR)
DATA_DIR_PROCESSED = DATA_DIR/"processed"/STAGE

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)   

RANDOM_STATE=41
fix_seed(RANDOM_STATE)

def r2_lgb(y_pred, dtrain):
    is_higher_better=True
    y_true = dtrain.label
    loss = skm.r2_score(y_true,y_pred)
    return 'r2', loss, is_higher_better


FEAT = {}
FEAT[0] = ['AOD_MODEL_mean', 'AOD_MODEL_mean_loc', 'AOD_MODEL_median', 'AOD_QA_mean', 'AOD_QA_mean_loc', 'AOD_QA_median', 'AOD_Uncertainty_mean', 'AOD_Uncertainty_mean_loc', 'AOD_Uncertainty_median', 'Column_WV_mean', 'Column_WV_mean_loc', 'Column_WV_median', 'Glint_Angle_mean', 'Glint_Angle_mean_loc', 'Glint_Angle_median', 'Injection_Height_mean', 'Injection_Height_mean_loc', 'Injection_Height_median', 'Optical_Depth_047_mean', 'Optical_Depth_047_mean_loc', 'Optical_Depth_047_median', 'Optical_Depth_055_mean', 'Optical_Depth_055_mean_loc', 'Optical_Depth_055_median', 'RelAZ_mean', 'RelAZ_mean_loc', 'RelAZ_median', 'Scattering_Angle_mean', 'Scattering_Angle_mean_loc', 'Scattering_Angle_median', 'absv_1000_mean', 'absv_100_mean', 'absv_10_mean', 'absv_150_mean', 'absv_200_mean', 'absv_20_mean', 'absv_250_mean', 'absv_300_mean', 'absv_350_mean', 'absv_400_mean', 'absv_450_mean', 'absv_500_mean', 'absv_50_mean', 'absv_550_mean', 'absv_600_mean', 'absv_650_mean', 'absv_700_mean', 'absv_750_mean', 'absv_800_mean', 'absv_850_mean', 'absv_900_mean', 'absv_925_mean', 'absv_950_mean', 'absv_975_mean', 'aptmp_max', 'aptmp_mean', 'aptmp_min', 'cape_18000_mean', 'cape_25500_mean', 'cin_18000_mean', 'cin_25500_mean', 'clwmr_100_mean', 'clwmr_150_mean', 'clwmr_200_mean', 'clwmr_250_mean', 'clwmr_300_mean', 'clwmr_350_mean', 'clwmr_400_mean', 'clwmr_450_mean', 'clwmr_500_mean', 'clwmr_550_mean', 'clwmr_600_mean', 'clwmr_650_mean', 'clwmr_700_mean', 'clwmr_750_mean', 'clwmr_800_mean', 'clwmr_850_mean', 'clwmr_900_mean', 'clwmr_925_mean', 'clwmr_950_mean', 'clwmr_975_mean', 'cosSZA_mean', 'cosSZA_mean_loc', 'cosSZA_median', 'cosVZA_mean', 'cosVZA_mean_loc', 'cosVZA_median', 'dpt_max', 'dpt_mean', 'dpt_min', 'elev_kurt', 'elev_max', 'elev_mean', 'elev_median', 'elev_min', 'elev_skew', 'elev_std', 'grid_max_bck_az_25', 'grid_max_dist_25', 'grid_max_fwd_az_25', 'grid_maxx', 'grid_maxy', 'grid_min_bck_az_25', 'grid_min_dist_25', 'grid_min_fwd_az_25', 'grid_minx', 'grid_miny', 'gust_max', 'gust_mean', 'gust_min', 'hgt_1000_mean', 'hgt_100_mean', 'hgt_10_mean', 'hgt_150_mean', 'hgt_1_mean', 'hgt_200_mean', 'hgt_20_mean', 'hgt_250_mean', 'hgt_300_mean', 'hgt_350_mean', 'hgt_400_mean', 'hgt_450_mean', 'hgt_500_mean', 'hgt_50_mean', 'hgt_550_mean', 'hgt_5_mean', 'hgt_600_mean', 'hgt_650_mean', 'hgt_700_mean', 'hgt_750_mean', 'hgt_800_mean', 'hgt_850_mean', 'hgt_900_mean', 'hgt_925_mean', 'hgt_950_mean', 'hgt_975_mean', 'hpbl_max', 'hpbl_mean', 'hpbl_min', 'latitude', 'longitude', 'nhour', 'o3mr_100_mean', 'o3mr_10_mean', 'o3mr_150_mean', 'o3mr_1_mean', 'o3mr_200_mean', 'o3mr_20_mean', 'o3mr_250_mean', 'o3mr_300_mean', 'o3mr_350_mean', 'o3mr_400_mean', 'o3mr_50_mean', 'o3mr_5_mean', 'obs_datetime_start_dow', 'obs_datetime_start_month', 'pevpr_max', 'pevpr_mean', 'pevpr_min', 'pot_max', 'pot_mean', 'pot_min', 'pres_max', 'pres_mean', 'pres_min', 'prmsl_max', 'prmsl_mean', 'prmsl_min', 'pwat_max', 'pwat_mean', 'pwat_min', 'rh_1000_mean', 'rh_100_mean', 'rh_10_mean', 'rh_150_mean', 'rh_1_mean', 'rh_200_mean', 'rh_20_mean', 'rh_250_mean', 'rh_300_mean', 'rh_350_mean', 'rh_400_mean', 'rh_450_mean', 'rh_500_mean', 'rh_50_mean', 'rh_550_mean', 'rh_5_mean', 'rh_600_mean', 'rh_650_mean', 'rh_700_mean', 'rh_750_mean', 'rh_800_mean', 'rh_850_mean', 'rh_900_mean', 'rh_925_mean', 'rh_950_mean', 'rh_975_mean', 'soilw_0.1_mean', 'soilw_0.4_mean', 'soilw_0_mean', 'soilw_1_mean', 'sunsd_max', 'sunsd_mean', 'tmp_1000_mean', 'tmp_100_mean', 'tmp_10_mean', 'tmp_150_mean', 'tmp_1_mean', 'tmp_200_mean', 'tmp_20_mean', 'tmp_250_mean', 'tmp_300_mean', 'tmp_350_mean', 'tmp_400_mean', 'tmp_450_mean', 'tmp_500_mean', 'tmp_50_mean', 'tmp_550_mean', 'tmp_5_mean', 'tmp_600_mean', 'tmp_650_mean', 'tmp_700_mean', 'tmp_750_mean', 'tmp_800_mean', 'tmp_850_mean', 'tmp_900_mean', 'tmp_925_mean', 'tmp_950_mean', 'tmp_975_mean', 'tozne_max', 'tozne_mean', 'tozne_min', 'tsoil_0.1_mean', 'tsoil_0.4_mean', 'tsoil_0_mean', 'tsoil_1_mean', 'ugrd_1000_mean', 'ugrd_100_mean', 'ugrd_10_mean', 'ugrd_150_mean', 'ugrd_1_mean', 'ugrd_200_mean', 'ugrd_20_mean', 'ugrd_250_mean', 'ugrd_300_mean', 'ugrd_350_mean', 'ugrd_400_mean', 'ugrd_450_mean', 'ugrd_500_mean', 'ugrd_50_mean', 'ugrd_550_mean', 'ugrd_5_mean', 'ugrd_600_mean', 'ugrd_650_mean', 'ugrd_700_mean', 'ugrd_750_mean', 'ugrd_800_mean', 'ugrd_850_mean', 'ugrd_900_mean', 'ugrd_925_mean', 'ugrd_950_mean', 'ugrd_975_mean', 'ustm_max', 'ustm_mean', 'ustm_min', 'vgrd_1000_mean', 'vgrd_100_mean', 'vgrd_10_mean', 'vgrd_150_mean', 'vgrd_1_mean', 'vgrd_200_mean', 'vgrd_20_mean', 'vgrd_250_mean', 'vgrd_300_mean', 'vgrd_350_mean', 'vgrd_400_mean', 'vgrd_450_mean', 'vgrd_500_mean', 'vgrd_50_mean', 'vgrd_550_mean', 'vgrd_5_mean', 'vgrd_600_mean', 'vgrd_650_mean', 'vgrd_700_mean', 'vgrd_750_mean', 'vgrd_800_mean', 'vgrd_850_mean', 'vgrd_900_mean', 'vgrd_925_mean', 'vgrd_950_mean', 'vgrd_975_mean', 'vrate_max', 'vrate_mean', 'vrate_min', 'vstm_max', 'vstm_mean', 'vstm_min', 'vwsh_2_mean']
FEAT[1] = ['AOD_QA_mean', 'AOD_QA_mean_loc', 'AOD_QA_median', 'AOD_Uncertainty_mean', 'AOD_Uncertainty_mean_loc', 'AOD_Uncertainty_median', 'Column_WV_mean', 'Column_WV_mean_loc', 'Column_WV_median', 'Glint_Angle_mean', 'Glint_Angle_mean_loc', 'Glint_Angle_median', 'Injection_Height_mean', 'Injection_Height_mean_loc', 'Injection_Height_median', 'Optical_Depth_047_mean', 'Optical_Depth_047_mean_loc', 'Optical_Depth_047_median', 'Optical_Depth_055_mean', 'Optical_Depth_055_mean_loc', 'Optical_Depth_055_median', 'RelAZ_mean', 'RelAZ_mean_loc', 'RelAZ_median', 'Scattering_Angle_mean', 'Scattering_Angle_mean_loc', 'Scattering_Angle_median', 'absv_1000_mean', 'absv_100_mean', 'absv_10_mean', 'absv_150_mean', 'absv_200_mean', 'absv_20_mean', 'absv_250_mean', 'absv_300_mean', 'absv_350_mean', 'absv_400_mean', 'absv_450_mean', 'absv_500_mean', 'absv_50_mean', 'absv_550_mean', 'absv_600_mean', 'absv_650_mean', 'absv_700_mean', 'absv_750_mean', 'absv_800_mean', 'absv_850_mean', 'absv_900_mean', 'absv_925_mean', 'absv_950_mean', 'absv_975_mean', 'aptmp_max', 'aptmp_mean', 'aptmp_min', 'cape_18000_mean', 'cape_25500_mean', 'cin_18000_mean', 'cin_25500_mean', 'clwmr_1000_mean', 'clwmr_100_mean', 'clwmr_150_mean', 'clwmr_200_mean', 'clwmr_250_mean', 'clwmr_300_mean', 'clwmr_350_mean', 'clwmr_400_mean', 'clwmr_450_mean', 'clwmr_500_mean', 'clwmr_550_mean', 'clwmr_600_mean', 'clwmr_650_mean', 'clwmr_700_mean', 'clwmr_750_mean', 'clwmr_800_mean', 'clwmr_850_mean', 'clwmr_900_mean', 'clwmr_925_mean', 'clwmr_950_mean', 'clwmr_975_mean', 'cosSZA_mean', 'cosSZA_mean_loc', 'cosSZA_median', 'cosVZA_mean', 'cosVZA_mean_loc', 'cosVZA_median', 'dpt_max', 'dpt_mean', 'dpt_min', 'elev_kurt', 'elev_max', 'elev_mean', 'elev_median', 'elev_min', 'elev_skew', 'elev_std', 'grid_max_bck_az_25', 'grid_max_dist_25', 'grid_max_fwd_az_25', 'grid_maxx', 'grid_maxy', 'grid_min_bck_az_25', 'grid_min_dist_25', 'grid_min_fwd_az_25', 'grid_minx', 'grid_miny', 'gust_max', 'gust_mean', 'gust_min', 'hgt_1000_mean', 'hgt_100_mean', 'hgt_10_mean', 'hgt_150_mean', 'hgt_1_mean', 'hgt_200_mean', 'hgt_20_mean', 'hgt_250_mean', 'hgt_300_mean', 'hgt_350_mean', 'hgt_400_mean', 'hgt_450_mean', 'hgt_500_mean', 'hgt_50_mean', 'hgt_550_mean', 'hgt_5_mean', 'hgt_600_mean', 'hgt_650_mean', 'hgt_700_mean', 'hgt_750_mean', 'hgt_800_mean', 'hgt_850_mean', 'hgt_900_mean', 'hgt_925_mean', 'hgt_950_mean', 'hgt_975_mean', 'hpbl_max', 'hpbl_mean', 'hpbl_min', 'latitude', 'longitude', 'nhour', 'o3mr_100_mean', 'o3mr_10_mean', 'o3mr_150_mean', 'o3mr_1_mean', 'o3mr_200_mean', 'o3mr_20_mean', 'o3mr_250_mean', 'o3mr_300_mean', 'o3mr_350_mean', 'o3mr_400_mean', 'o3mr_50_mean', 'o3mr_5_mean', 'obs_datetime_start_dow', 'obs_datetime_start_month', 'pevpr_max', 'pevpr_mean', 'pevpr_min', 'pot_max', 'pot_mean', 'pot_min', 'pres_max', 'pres_mean', 'pres_min', 'prmsl_max', 'prmsl_mean', 'prmsl_min', 'pwat_max', 'pwat_mean', 'pwat_min', 'rh_1000_mean', 'rh_100_mean', 'rh_10_mean', 'rh_150_mean', 'rh_1_mean', 'rh_200_mean', 'rh_20_mean', 'rh_250_mean', 'rh_300_mean', 'rh_350_mean', 'rh_400_mean', 'rh_450_mean', 'rh_500_mean', 'rh_50_mean', 'rh_550_mean', 'rh_5_mean', 'rh_600_mean', 'rh_650_mean', 'rh_700_mean', 'rh_750_mean', 'rh_800_mean', 'rh_850_mean', 'rh_900_mean', 'rh_925_mean', 'rh_950_mean', 'rh_975_mean', 'soilw_0.1_mean', 'soilw_0.4_mean', 'soilw_0_mean', 'soilw_1_mean', 'sunsd_max', 'sunsd_mean', 'tmp_1000_mean', 'tmp_100_mean', 'tmp_10_mean', 'tmp_150_mean', 'tmp_1_mean', 'tmp_200_mean', 'tmp_20_mean', 'tmp_250_mean', 'tmp_300_mean', 'tmp_350_mean', 'tmp_400_mean', 'tmp_450_mean', 'tmp_500_mean', 'tmp_50_mean', 'tmp_550_mean', 'tmp_5_mean', 'tmp_600_mean', 'tmp_650_mean', 'tmp_700_mean', 'tmp_750_mean', 'tmp_800_mean', 'tmp_850_mean', 'tmp_900_mean', 'tmp_925_mean', 'tmp_950_mean', 'tmp_975_mean', 'tozne_max', 'tozne_mean', 'tozne_min', 'tsoil_0.1_mean', 'tsoil_0.4_mean', 'tsoil_0_mean', 'tsoil_1_mean', 'ugrd_1000_mean', 'ugrd_100_mean', 'ugrd_10_mean', 'ugrd_150_mean', 'ugrd_1_mean', 'ugrd_200_mean', 'ugrd_20_mean', 'ugrd_250_mean', 'ugrd_300_mean', 'ugrd_350_mean', 'ugrd_400_mean', 'ugrd_450_mean', 'ugrd_500_mean', 'ugrd_50_mean', 'ugrd_550_mean', 'ugrd_5_mean', 'ugrd_600_mean', 'ugrd_650_mean', 'ugrd_700_mean', 'ugrd_750_mean', 'ugrd_800_mean', 'ugrd_850_mean', 'ugrd_900_mean', 'ugrd_925_mean', 'ugrd_950_mean', 'ugrd_975_mean', 'ustm_max', 'ustm_mean', 'ustm_min', 'vgrd_1000_mean', 'vgrd_100_mean', 'vgrd_10_mean', 'vgrd_150_mean', 'vgrd_1_mean', 'vgrd_200_mean', 'vgrd_20_mean', 'vgrd_250_mean', 'vgrd_300_mean', 'vgrd_350_mean', 'vgrd_400_mean', 'vgrd_450_mean', 'vgrd_500_mean', 'vgrd_50_mean', 'vgrd_550_mean', 'vgrd_5_mean', 'vgrd_600_mean', 'vgrd_650_mean', 'vgrd_700_mean', 'vgrd_750_mean', 'vgrd_800_mean', 'vgrd_850_mean', 'vgrd_900_mean', 'vgrd_925_mean', 'vgrd_950_mean', 'vgrd_975_mean', 'vrate_max', 'vrate_mean', 'vrate_min', 'vstm_max', 'vstm_mean', 'vstm_min', 'vwsh_2_mean']
FEAT[2] = ['AOD_QA_mean', 'AOD_QA_mean_loc', 'AOD_QA_median', 'AOD_Uncertainty_mean', 'AOD_Uncertainty_mean_loc', 'AOD_Uncertainty_median', 'Column_WV_mean', 'Column_WV_mean_loc', 'Column_WV_median', 'Glint_Angle_mean', 'Glint_Angle_mean_loc', 'Glint_Angle_median', 'Injection_Height_mean', 'Injection_Height_mean_loc', 'Injection_Height_median', 'Optical_Depth_047_mean', 'Optical_Depth_047_mean_loc', 'Optical_Depth_047_median', 'Optical_Depth_055_mean', 'Optical_Depth_055_mean_loc', 'Optical_Depth_055_median', 'RelAZ_mean', 'RelAZ_mean_loc', 'RelAZ_median', 'Scattering_Angle_mean', 'Scattering_Angle_mean_loc', 'Scattering_Angle_median', 'absv_1000_mean', 'absv_100_mean', 'absv_10_mean', 'absv_150_mean', 'absv_200_mean', 'absv_20_mean', 'absv_250_mean', 'absv_300_mean', 'absv_350_mean', 'absv_400_mean', 'absv_450_mean', 'absv_500_mean', 'absv_50_mean', 'absv_550_mean', 'absv_600_mean', 'absv_650_mean', 'absv_700_mean', 'absv_750_mean', 'absv_800_mean', 'absv_850_mean', 'absv_900_mean', 'absv_925_mean', 'absv_950_mean', 'absv_975_mean', 'aptmp_max', 'aptmp_mean', 'aptmp_min', 'cape_18000_mean', 'cape_25500_mean', 'cin_18000_mean', 'cin_25500_mean', 'clwmr_1000_mean', 'clwmr_100_mean', 'clwmr_150_mean', 'clwmr_200_mean', 'clwmr_250_mean', 'clwmr_300_mean', 'clwmr_350_mean', 'clwmr_400_mean', 'clwmr_450_mean', 'clwmr_500_mean', 'clwmr_550_mean', 'clwmr_600_mean', 'clwmr_650_mean', 'clwmr_700_mean', 'clwmr_750_mean', 'clwmr_800_mean', 'clwmr_850_mean', 'clwmr_900_mean', 'clwmr_925_mean', 'clwmr_950_mean', 'clwmr_975_mean', 'cosSZA_mean', 'cosSZA_mean_loc', 'cosSZA_median', 'cosVZA_mean', 'cosVZA_mean_loc', 'cosVZA_median', 'dpt_max', 'dpt_mean', 'dpt_min', 'elev_kurt', 'elev_max', 'elev_mean', 'elev_median', 'elev_min', 'elev_skew', 'elev_std', 'grid_max_bck_az_25', 'grid_max_dist_25', 'grid_max_fwd_az_25', 'grid_maxx', 'grid_maxy', 'grid_min_bck_az_25', 'grid_min_dist_25', 'grid_min_fwd_az_25', 'grid_minx', 'grid_miny', 'gust_max', 'gust_mean', 'gust_min', 'hgt_1000_mean', 'hgt_100_mean', 'hgt_10_mean', 'hgt_150_mean', 'hgt_1_mean', 'hgt_200_mean', 'hgt_20_mean', 'hgt_250_mean', 'hgt_300_mean', 'hgt_350_mean', 'hgt_400_mean', 'hgt_450_mean', 'hgt_500_mean', 'hgt_50_mean', 'hgt_550_mean', 'hgt_5_mean', 'hgt_600_mean', 'hgt_650_mean', 'hgt_700_mean', 'hgt_750_mean', 'hgt_800_mean', 'hgt_850_mean', 'hgt_900_mean', 'hgt_925_mean', 'hgt_950_mean', 'hgt_975_mean', 'hpbl_max', 'hpbl_mean', 'hpbl_min', 'nhour', 'o3mr_100_mean', 'o3mr_10_mean', 'o3mr_150_mean', 'o3mr_1_mean', 'o3mr_200_mean', 'o3mr_20_mean', 'o3mr_250_mean', 'o3mr_300_mean', 'o3mr_350_mean', 'o3mr_400_mean', 'o3mr_50_mean', 'o3mr_5_mean', 'obs_datetime_start_dow', 'obs_datetime_start_month', 'pevpr_max', 'pevpr_mean', 'pevpr_min', 'pot_max', 'pot_mean', 'pot_min', 'pres_max', 'pres_mean', 'pres_min', 'prmsl_max', 'prmsl_mean', 'prmsl_min', 'pwat_max', 'pwat_mean', 'pwat_min', 'rh_1000_mean', 'rh_100_mean', 'rh_10_mean', 'rh_150_mean', 'rh_1_mean', 'rh_200_mean', 'rh_20_mean', 'rh_250_mean', 'rh_300_mean', 'rh_350_mean', 'rh_400_mean', 'rh_450_mean', 'rh_500_mean', 'rh_50_mean', 'rh_550_mean', 'rh_5_mean', 'rh_600_mean', 'rh_650_mean', 'rh_700_mean', 'rh_750_mean', 'rh_800_mean', 'rh_850_mean', 'rh_900_mean', 'rh_925_mean', 'rh_950_mean', 'rh_975_mean', 'soilw_0.1_mean', 'soilw_0.4_mean', 'soilw_0_mean', 'soilw_1_mean', 'sunsd_max', 'sunsd_mean', 'tmp_1000_mean', 'tmp_100_mean', 'tmp_10_mean', 'tmp_150_mean', 'tmp_1_mean', 'tmp_200_mean', 'tmp_20_mean', 'tmp_250_mean', 'tmp_300_mean', 'tmp_350_mean', 'tmp_400_mean', 'tmp_450_mean', 'tmp_500_mean', 'tmp_50_mean', 'tmp_550_mean', 'tmp_5_mean', 'tmp_600_mean', 'tmp_650_mean', 'tmp_700_mean', 'tmp_750_mean', 'tmp_800_mean', 'tmp_850_mean', 'tmp_900_mean', 'tmp_925_mean', 'tmp_950_mean', 'tmp_975_mean', 'tozne_max', 'tozne_mean', 'tozne_min', 'tsoil_0.1_mean', 'tsoil_0.4_mean', 'tsoil_0_mean', 'tsoil_1_mean', 'ugrd_1000_mean', 'ugrd_100_mean', 'ugrd_10_mean', 'ugrd_150_mean', 'ugrd_1_mean', 'ugrd_200_mean', 'ugrd_20_mean', 'ugrd_250_mean', 'ugrd_300_mean', 'ugrd_350_mean', 'ugrd_400_mean', 'ugrd_450_mean', 'ugrd_500_mean', 'ugrd_50_mean', 'ugrd_550_mean', 'ugrd_5_mean', 'ugrd_600_mean', 'ugrd_650_mean', 'ugrd_700_mean', 'ugrd_750_mean', 'ugrd_800_mean', 'ugrd_850_mean', 'ugrd_900_mean', 'ugrd_925_mean', 'ugrd_950_mean', 'ugrd_975_mean', 'ustm_max', 'ustm_mean', 'ustm_min', 'vgrd_1000_mean', 'vgrd_100_mean', 'vgrd_10_mean', 'vgrd_150_mean', 'vgrd_1_mean', 'vgrd_200_mean', 'vgrd_20_mean', 'vgrd_250_mean', 'vgrd_300_mean', 'vgrd_350_mean', 'vgrd_400_mean', 'vgrd_450_mean', 'vgrd_500_mean', 'vgrd_50_mean', 'vgrd_550_mean', 'vgrd_5_mean', 'vgrd_600_mean', 'vgrd_650_mean', 'vgrd_700_mean', 'vgrd_750_mean', 'vgrd_800_mean', 'vgrd_850_mean', 'vgrd_900_mean', 'vgrd_925_mean', 'vgrd_950_mean', 'vgrd_975_mean', 'vrate_max', 'vrate_mean', 'vrate_min', 'vstm_max', 'vstm_mean', 'vstm_min', 'vwsh_2_mean']
sort_cols = ['datetime','grid_id']
N_SPLITS=5


def get_data(dsid):
    df_data = pd.read_pickle(f'{DATA_DIR_PROCESSED}/{STAGE}_tail{dsid}.pkl') 
    df_data = df_data.sort_values(by=sort_cols).reset_index(drop=True)
    return df_data

def get_data_for_loc(df,location_code):
    features = FEAT[location_code]
    df = df[df.location_code==location_code].copy()
    df_ref = df[['location_code','grid_id','datetime','fold','target']].copy()
    X = df[features].copy()
    y = df.target.values
    return X,y,df_ref



def evaluate(dsid):
    df_test = get_data(dsid)
    res_test = []
    df_test['fold']=-1
    for location_code in df_test.location_code.astype(int).unique():
        X_test,y_test,ref_test = get_data_for_loc(df_test,location_code)
        for fold in range(N_SPLITS):
            model_name = f'ds{dsid}_loc{location_code}_f{fold}'
            model = lgb.Booster(model_file=f'{MODEL_DIR}/{model_name}')
            y_test_preds = model.predict(X_test)
            ref_test[f'pred_{fold}'] = y_test_preds
            print(f'location {location_code}, fold: {fold}, samples:{y_test_preds.shape[0]} {y_test_preds.min()},{y_test_preds.max()}' )
        res_test.append(ref_test)
    df_res = pd.concat(res_test)
    pred_cols = [f'pred_{i}' for i in range(N_SPLITS)] 
    df_res['value'] = df_res[pred_cols].mean(axis=1)
    
    return df_res

RES_TEST = []
for dsid in [8,10,12]:
    print(f'##### RUNNING INFERENCE FOR DATASET {dsid} ##########')
    df_res = evaluate(dsid)
    RES_TEST.append(df_res)

sub = RES_TEST[0][sort_cols].copy()
s=np.mean([RES_TEST[0]['value'].values,RES_TEST[1]['value'].values,RES_TEST[2]['value'].values],axis=0)
sub['value'] = s
sub = sub.set_index(sort_cols)
print('predictions range: ',s.min(),s.max())
submission_format = pd.read_csv(SUB_FORMAT).set_index(sort_cols)
sub = sub.reindex(submission_format.index).reset_index()
sub = sub[['datetime','grid_id','value']]
sub.to_csv(OUTPUT_PATH,index=False)
print(f'Written predictions to {OUTPUT_PATH}')