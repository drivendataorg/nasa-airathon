import os

def download_s3_file(s3_path,local_path):
    if not os.path.exists(local_path):
        cmd = f'aws s3 cp {s3_path} {local_path} --no-sign-request --quiet'
        ret = os.system(cmd)
        if ret!=0:
            raise Exception(f'Failed to download {s3_path} to {local_path}. System returned {ret}')
