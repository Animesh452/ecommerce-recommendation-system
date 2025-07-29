# etl/setup-s3-data-lake.py
# Setup script to create S3 data lake structure and upload raw data

import boto3
import os
from datetime import datetime

print("🏗️ Setting up S3 Data Lake for E-commerce Recommendation System")

# Configuration - UPDATE WITH YOUR BUCKET NAME
BUCKET_NAME = "ecommerce-recommendation-data-lake"  # Change this to your bucket name
REGION = "us-east-1"  # Change to your preferred region

def create_s3_bucket():
    """Create S3 bucket for data lake"""
    
    s3_client = boto3.client('s3', region_name=REGION)
    
    try:
        if REGION == 'us-east-1':
            s3_client.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3_client.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={'LocationConstraint': REGION}
            )
        
        print(f"✅ Created S3 bucket: {BUCKET_NAME}")
        return True
        
    except Exception as e:
        if "BucketAlreadyOwnedByYou" in str(e):
            print(f"✅ S3 bucket already exists: {BUCKET_NAME}")
            return True
        else:
            print(f"❌ Error creating bucket: {e}")
            return False

def create_data_lake_structure():
    """Create folder structure in S3 data lake"""
    
    s3_client = boto3.client('s3')
    
    folders = [
        "ecommerce-data/raw/",
        "ecommerce-data/processed/collaborative_filtering/",
        "ecommerce-data/processed/content_based/",
        "ecommerce-data/processed/metadata/",
        "ml-models/collaborative-filtering/",
        "ml-models/content-based/",
        "ml-models/hybrid/",
        "api-data/predictions/",
        "monitoring/metrics/"
    ]
    
    for folder in folders:
        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=folder,
                Body=''
            )
            print(f"📁 Created folder: s3://{BUCKET_NAME}/{folder}")
        except Exception as e:
            print(f"⚠️ Could not create folder {folder}: {e}")

def upload_raw_data():
    """Upload generated CSV files to S3 raw data folder"""
    
    s3_client = boto3.client('s3')
    
    # Files to upload
    data_files = [
        'data/raw/products.csv',
        'data/raw/users.csv', 
        'data/raw/interactions.csv'
    ]
    
    for local_file in data_files:
        if os.path.exists(local_file):
            s3_key = f"ecommerce-data/raw/{os.path.basename(local_file)}"
            
            try:
                s3_client.upload_file(local_file, BUCKET_NAME, s3_key)
                print(f"📤 Uploaded: {local_file} → s3://{BUCKET_NAME}/{s3_key}")
            except Exception as e:
                print(f"❌ Error uploading {local_file}: {e}")
        else:
            print(f"⚠️ File not found: {local_file}")
            print("   Run 'python data-generation.py' first to create the data files")

def update_glue_script_config():
    """Update the Glue script with correct S3 paths"""
    
    glue_script_path = "etl/glue-jobs/ecommerce-data-processing.py"
    
    if os.path.exists(glue_script_path):
        with open(glue_script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update S3 paths
        content = content.replace(
            'S3_RAW_DATA_PATH = "s3://your-bucket-name/ecommerce-data/raw/"',
            f'S3_RAW_DATA_PATH = "s3://{BUCKET_NAME}/ecommerce-data/raw/"'
        )
        content = content.replace(
            'S3_PROCESSED_DATA_PATH = "s3://your-bucket-name/ecommerce-data/processed/"',
            f'S3_PROCESSED_DATA_PATH = "s3://{BUCKET_NAME}/ecommerce-data/processed/"'
        )
        
        with open(glue_script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Updated Glue script with bucket name: {BUCKET_NAME}")
    else:
        print(f"⚠️ Glue script not found: {glue_script_path}")

def main():
    """Run the complete setup process"""
    
    print(f"🎯 Setting up data lake with bucket: {BUCKET_NAME}")
    print(f"📍 Region: {REGION}")
    
    # Step 1: Create S3 bucket
    if create_s3_bucket():
        # Step 2: Create folder structure
        create_data_lake_structure()
        
        # Step 3: Upload raw data
        upload_raw_data()
        
        # Step 4: Update Glue script configuration
        update_glue_script_config()
        
        print(f"""
🎉 DATA LAKE SETUP COMPLETE!

📊 Your S3 Data Lake:
   Bucket: s3://{BUCKET_NAME}
   Region: {REGION}
        """)
    else:
        print("❌ Setup failed. Please check your AWS credentials and permissions.")

if __name__ == "__main__":
    main()