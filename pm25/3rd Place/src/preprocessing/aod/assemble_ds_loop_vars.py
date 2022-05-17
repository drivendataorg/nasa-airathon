import os

split_type = 'test'
datafs = ['Optical_Depth_055','Optical_Depth_047','AOD_Uncertainty','FineModeFraction','Column_WV','AOD_QA','AOD_MODEL','Injection_Height']
extracted_vars_path = f'extracted_vars/{split_type}/maiac/'
save_path = f'assembled_csv/{split_type}'
for arg in datafs:
  os.system("python assemble_dataset.py {} {} {}".format(extracted_vars_path, save_path, arg))