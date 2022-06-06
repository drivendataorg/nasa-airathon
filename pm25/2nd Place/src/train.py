import pandas as pd, numpy as np
import os,sys,random,argparse
from pathlib import Path
from sklearn.model_selection import KFold,TimeSeriesSplit
import sklearn.metrics as skm
import lightgbm as lgb


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Path to root data directory", required=True)
parser.add_argument("--model_dir", help="Directory to save models", required=True, default='models')

args = parser.parse_args()
DATA_DIR = args.data_dir 
MODEL_DIR = args.model_dir 

# DATA_DIR = 'data'
# MODEL_DIR = 'models'

DATA_DIR = Path(DATA_DIR) 
MODEL_DIR = Path(MODEL_DIR)
DATA_DIR_RAW = DATA_DIR / "raw"
DATA_DIR_PROCESSED = DATA_DIR/"processed/train"
os.makedirs(MODEL_DIR,exist_ok=True)

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

ALL_PARAMS = {}
PARAMS8 = {}
PARAMS8[0]={'boost_from_average': 'false', 'max_depth': 5, 'num_leaves': 100, 'objective': 'rmse', 'metric': 'rmse', 'boosting': 'gbdt', 'bagging_seed': RANDOM_STATE, 'seed': RANDOM_STATE, 'verbosity': -1, 'num_threads': 4, 'learning_rate': 0.1}	
PARAMS8[1]={'boost_from_average': 'false', 'max_depth': 8, 'num_leaves': 64, 'objective': 'rmse', 'metric': 'rmse', 'boosting': 'gbdt', 'bagging_seed': RANDOM_STATE, 'seed': RANDOM_STATE, 'verbosity': -1, 'num_threads': 4, 'learning_rate': 0.1}	
PARAMS8[2]={'boost_from_average': 'true', 'max_depth': 6, 'num_leaves': 17, 'objective': 'rmse', 'metric': 'rmse', 'boosting': 'gbdt', 'bagging_seed': RANDOM_STATE, 'seed': RANDOM_STATE, 'verbosity': -1, 'num_threads': 4, 'learning_rate': 0.1}	
ALL_PARAMS[8] = PARAMS8

PARAMS10 = {}
PARAMS10[0]={'boost_from_average': 'false', 'max_depth': 3, 'num_leaves': 100, 'objective': 'rmse', 'metric': 'rmse', 'boosting': 'gbdt', 'bagging_seed': RANDOM_STATE, 'seed': RANDOM_STATE, 'verbosity': -1, 'num_threads': 4, 'learning_rate': 0.1}	
PARAMS10[1]={'boost_from_average': 'false', 'max_depth': 5, 'num_leaves': 28, 'objective': 'rmse', 'metric': 'rmse', 'boosting': 'gbdt', 'bagging_seed': RANDOM_STATE, 'seed': RANDOM_STATE, 'verbosity': -1, 'num_threads': 4, 'learning_rate': 0.1}	
PARAMS10[2]={'boost_from_average': 'true', 'max_depth': 3, 'num_leaves': 100, 'objective': 'rmse', 'metric': 'rmse', 'boosting': 'gbdt', 'bagging_seed': RANDOM_STATE, 'seed': RANDOM_STATE, 'verbosity': -1, 'num_threads': 4, 'learning_rate': 0.1}	
ALL_PARAMS[10] = PARAMS10

PARAMS12 = {}
PARAMS12[0] = {'boost_from_average': 'false', 'max_depth': 5, 'num_leaves': 68, 'objective': 'rmse', 'metric': 'rmse', 'boosting': 'gbdt', 'bagging_seed': RANDOM_STATE, 'seed': RANDOM_STATE, 'verbosity': -1, 'num_threads': 4, 'learning_rate': 0.1}	
PARAMS12[1] = {'boost_from_average': 'false', 'max_depth': 11, 'num_leaves': 40, 'objective': 'rmse', 'metric': 'rmse', 'boosting': 'gbdt', 'bagging_seed': RANDOM_STATE, 'seed': RANDOM_STATE, 'verbosity': -1, 'num_threads': 4, 'learning_rate': 0.1}	
PARAMS12[2]= {'num_leaves': 33, 'objective': 'huber', 'metric': 'rmse', 'boosting': 'gbdt', 'bagging_seed': RANDOM_STATE, 'seed': RANDOM_STATE, 'verbosity': -1, 'num_threads': 4, 'learning_rate': 0.1, 'max_depth': -1}	
ALL_PARAMS[12] = PARAMS12


N_SPLITS=5

def get_data(dsid):
    sort_cols = ['datetime','grid_id']
    df_data = pd.read_pickle(f'{DATA_DIR_PROCESSED}/train_tail{dsid}.pkl') 
    df_data = df_data.sort_values(by=sort_cols).reset_index(drop=True)
    data_cv = []
    for location_code in range(3):
        print(f'location {location_code}')
        df=  df_data[df_data.location_code==location_code].reset_index(drop=True)
        kf = KFold(n_splits=N_SPLITS,shuffle=False)
        df['fold'] = -1
        for i,(train_index, val_index) in enumerate(kf.split(df)):
            #print('trn_val:', train_index.shape[0],val_index.shape[0])
            df.loc[val_index,'fold'] = i

        for f in range(1,N_SPLITS):
            df.loc[(df.fold==f)&(df.obs_datetime_start==df[df.fold==f-1].obs_datetime_start.max()),'fold'] = f-1

        data_cv.append(df)
    data_cv = pd.concat(data_cv).sort_values(by=sort_cols).reset_index(drop=True)
    print(data_cv.fold.nunique(),data_cv.fold.unique())
    return data_cv

def get_data_for_loc(df,location_code):
    features = FEAT[location_code]
    df = df[df.location_code==location_code].copy()
    df_ref = df[['location_code','grid_id','datetime','fold','target']].copy()
    X = df[features].copy()
    y = df.target.values
    return X,y,df_ref

def train_model_cb(location_code,params,trn,val):
    print(f'TRAINING MODEL FOR LOCATION {location_code}')
    X_train,y_train,ref_train = get_data_for_loc(trn,location_code)
    X_valid,y_valid,ref_val = get_data_for_loc(val,location_code)
    val_start_date = ref_val.datetime.min()
    val_end_date = ref_val.datetime.max()
    print(f'location {location_code} val range: ',val_start_date, val_end_date)
    eval_dataset = Pool(X_valid,y_valid)
    
    model = CatBoostRegressor(learning_rate=params['learning_rate'],depth=params['depth'],
                          task_type=params['task_type'],random_seed=params['seed'],
                           num_boost_round=100000,early_stopping_rounds=500)#
    model.fit(X_train,y_train,eval_set=eval_dataset,verbose=10000)

    y_valid_pred=model.predict(X_valid)
    print('range: ',y_valid_pred.min(),y_valid_pred.max())
    rms,r2 = skm.mean_squared_error(y_valid,y_valid_pred,squared=False),skm.r2_score(y_valid,y_valid_pred)
    print(f'location: {location_code}, rmse:{rms}, r2: {r2}')
    ref_val['pred']=y_valid_pred
    return model ,ref_val


def train_model_lgb(location_code,params,trn,val):
    print(f'TRAINING MODEL FOR LOCATION {location_code}')
    X_train,y_train,ref_train = get_data_for_loc(trn,location_code)
    X_valid,y_valid,ref_val = get_data_for_loc(val,location_code)
    val_start_date = ref_val.datetime.min()
    val_end_date = ref_val.datetime.max()
    print(f'location {location_code} val range: ',val_start_date, val_end_date)
    lgb_train = lgb.Dataset(X_train, y_train,free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train,free_raw_data=False)
    evals_result = {} 
    model = lgb.train(params, lgb_train, valid_sets=[lgb_train,lgb_valid], valid_names=['train','valid'],
                  evals_result=evals_result,feval=r2_lgb,early_stopping_rounds=500, num_boost_round=100000,verbose_eval=10000)
    y_valid_pred=model.predict(X_valid)
    print('range: ',y_valid_pred.min(),y_valid_pred.max())
    rms,r2 = skm.mean_squared_error(y_valid,y_valid_pred,squared=False),skm.r2_score(y_valid,y_valid_pred)
    print(f'location: {location_code}, rmse:{rms}, r2: {r2}')
    ref_val['pred']=y_valid_pred
    return model ,ref_val, X_valid

def train(dsid):
    data_cv = get_data(dsid)
    PARAMS = ALL_PARAMS[dsid]
    REFS_VAL = {}
    X_VALID = {}
    for fold in range(N_SPLITS):
        df_val = data_cv[(data_cv.fold==fold)].copy()
        df_trn = data_cv[(data_cv.fold!=fold)].copy()

        val_start_date,val_end_date = str(df_val.obs_datetime_start.min()),str(df_val.obs_datetime_start.max())
        print(f'FOLD {fold} ALL LOCATIONS VAL RANGE: ',val_start_date, val_end_date)
        for location_code in range(3):
                params = PARAMS[location_code]
                model,ref_val,X_valid = train_model_lgb(location_code,params,trn=df_trn,val=df_val)
                model_name = f'ds{dsid}_loc{location_code}_f{fold}'
                model.save_model(f'{MODEL_DIR}/{model_name}')
                REFS_VAL[model_name] = ref_val
                X_VALID[model_name] = X_valid
    
    ##OOF inference
    p=4
    dfs_oof = []
    for fold in range(N_SPLITS):
        df_oof = []
        for location_code in range(3):
            model_name = f'ds{dsid}_loc{location_code}_f{fold}'
            model = lgb.Booster(model_file=f'{MODEL_DIR}/{model_name}')
            d = REFS_VAL[model_name]
            X_valid = X_VALID[model_name]
            y_pred = model.predict(X_valid)
            d['pred1'] = y_pred
            df_oof.append(d)
        df_oof = pd.concat(df_oof).reset_index(drop=True)
        dfs_oof.append(df_oof)
        rms,r2 = skm.mean_squared_error(df_oof.target,df_oof.pred,squared=False),skm.r2_score(df_oof.target,df_oof.pred)
        print(f'fold {fold}, {rms} {r2}')
    dfs_oof = pd.concat(dfs_oof).reset_index(drop=True)
    assert (dfs_oof.pred==dfs_oof.pred1).all()
    rms_all,r2_all = skm.mean_squared_error(dfs_oof.target,dfs_oof.pred,squared=False),skm.r2_score(dfs_oof.target,dfs_oof.pred)
    print(f'all folds, {rms_all} {r2_all}')
    locs_r2=[]
    for location_code in range(3):
        d = dfs_oof[dfs_oof.location_code==location_code]
        rms,r2 = skm.mean_squared_error(d.target,d.pred,squared=False),skm.r2_score(d.target,d.pred)
        print(f'loc {location_code}, {rms} {r2}')
        locs_r2.append(np.round(r2,p))

    rms_all=np.round(rms_all,p)
    r2_all=np.round(r2_all,p)
    print('cv score: ',rms_all,r2_all,locs_r2)

for dsid in [8,10,12]:
    print(f'##### TRAINING MODEL WITH DATASET {dsid} ##########')
    train(dsid)
