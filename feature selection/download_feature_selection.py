import boto3
import os

# AWS S3 Configuration
BUCKET_NAME = 's3-bucket-umairrr'
S3_KEY = 'feature-selection-multipipeline/feature_selection.csv'
LOCAL_PATH = os.path.join('data', 'feature_selection.csv')

def download_from_s3():
    s3 = boto3.client('s3')

    # Ensure local directory exists
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)

    try:
        s3.download_file(BUCKET_NAME, S3_KEY, LOCAL_PATH)
        print(f"✅ Downloaded feature_selection.csv to {LOCAL_PATH}")
    except Exception as e:
        print(f"❌ Failed to download: {e}")

if __name__ == "__main__":
    download_from_s3()
