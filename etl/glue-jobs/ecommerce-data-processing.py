# Day 1 Part 2: AWS Glue ETL Pipeline for Recommendation System
# Transform raw e-commerce data into ML-ready features

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import *
import datetime

print("🔄 Starting E-commerce Data ETL Pipeline")
print(f"Processing Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===== GLUE JOB SETUP =====
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

print("✅ Glue Context initialized")

# ===== CONFIGURATION =====
# Update these with your actual S3 bucket paths
S3_RAW_DATA_PATH = "s3://ecommerce-recommendation-data-lake/ecommerce-data/raw/"
S3_PROCESSED_DATA_PATH = "s3://ecommerce-recommendation-data-lake/ecommerce-data/processed/"

print(f"📍 Raw data path: {S3_RAW_DATA_PATH}")
print(f"📍 Processed data path: {S3_PROCESSED_DATA_PATH}")

# ===== PART 1: LOAD RAW DATA =====
print("\n=== 📥 Loading Raw E-commerce Data ===")

def load_raw_data():
    """Load raw CSV data from S3"""
    
    try:
        # Load products data
        products_df = spark.read.option("header", "true").option("inferSchema", "true").csv(
            S3_RAW_DATA_PATH + "products.csv"
        )
        print(f"✅ Loaded products: {products_df.count()} records")
        
        # Load users data
        users_df = spark.read.option("header", "true").option("inferSchema", "true").csv(
            S3_RAW_DATA_PATH + "users.csv"
        )
        print(f"✅ Loaded users: {users_df.count()} records")
        
        # Load interactions data
        interactions_df = spark.read.option("header", "true").option("inferSchema", "true").csv(
            S3_RAW_DATA_PATH + "interactions.csv"
        )
        print(f"✅ Loaded interactions: {interactions_df.count()} records")
        
        return products_df, users_df, interactions_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

products_df, users_df, interactions_df = load_raw_data()

if products_df is None:
    print("❌ Failed to load data. Exiting.")
    job.commit()
    sys.exit(1)

# ===== PART 2: DATA QUALITY CHECKS =====
print("\n=== 🔍 Data Quality Validation ===")

def perform_data_quality_checks(products_df, users_df, interactions_df):
    """Perform comprehensive data quality checks"""
    
    quality_issues = []
    
    # Products validation
    print("🔍 Validating products data...")
    products_count = products_df.count()
    products_null_count = products_df.filter(
        F.col("product_id").isNull() | 
        F.col("name").isNull() | 
        F.col("category").isNull()
    ).count()
    
    if products_null_count > 0:
        quality_issues.append(f"Products: {products_null_count} records with null key fields")
    else:
        print(f"  ✅ Products: {products_count} records, no null key fields")
    
    # Users validation
    print("🔍 Validating users data...")
    users_count = users_df.count()
    users_null_count = users_df.filter(F.col("user_id").isNull()).count()
    
    if users_null_count > 0:
        quality_issues.append(f"Users: {users_null_count} records with null user_id")
    else:
        print(f"  ✅ Users: {users_count} records, no null user IDs")
    
    # Interactions validation
    print("🔍 Validating interactions data...")
    interactions_count = interactions_df.count()
    interactions_null_count = interactions_df.filter(
        F.col("user_id").isNull() | 
        F.col("product_id").isNull()
    ).count()
    
    if interactions_null_count > 0:
        quality_issues.append(f"Interactions: {interactions_null_count} records with null IDs")
    else:
        print(f"  ✅ Interactions: {interactions_count} records, no null IDs")
    
    # Cross-validation: Check for orphaned records
    print("🔍 Cross-validating data integrity...")
    
    # Check for interactions with non-existent users
    user_ids = users_df.select("user_id").distinct()
    orphaned_users = interactions_df.join(user_ids, "user_id", "left_anti").count()
    if orphaned_users > 0:
        quality_issues.append(f"Interactions: {orphaned_users} records with non-existent users")
    
    # Check for interactions with non-existent products
    product_ids = products_df.select("product_id").distinct()
    orphaned_products = interactions_df.join(product_ids, "product_id", "left_anti").count()
    if orphaned_products > 0:
        quality_issues.append(f"Interactions: {orphaned_products} records with non-existent products")
    
    if not quality_issues:
        print("✅ All data quality checks passed!")
    else:
        print("⚠️ Data quality issues found:")
        for issue in quality_issues:
            print(f"  - {issue}")
    
    return len(quality_issues) == 0

data_quality_ok = perform_data_quality_checks(products_df, users_df, interactions_df)

# ===== PART 3: FEATURE ENGINEERING FOR COLLABORATIVE FILTERING =====
print("\n=== 🤝 Creating Collaborative Filtering Features ===")

def create_collaborative_filtering_features(interactions_df):
    """Create user-item interaction matrix and user/item features"""
    
    print("🔄 Building user-item interaction matrix...")
    
    # Create implicit ratings from interactions
    # Different interaction types get different weights
    interaction_weights = {
        'view': 1.0,
        'like': 2.0,  
        'add_to_cart': 3.0,
        'review': 4.0,
        'purchase': 5.0
    }
    
    # Add interaction weights
    interactions_weighted = interactions_df.withColumn(
        "interaction_weight",
        F.when(F.col("interaction_type") == "view", 1.0)
         .when(F.col("interaction_type") == "like", 2.0)
         .when(F.col("interaction_type") == "add_to_cart", 3.0)
         .when(F.col("interaction_type") == "review", 4.0)
         .when(F.col("interaction_type") == "purchase", 5.0)
         .otherwise(1.0)
    )
    
    # Aggregate interactions per user-item pair
    user_item_matrix = interactions_weighted.groupBy("user_id", "product_id").agg(
        F.sum("interaction_weight").alias("interaction_score"),
        F.count("*").alias("interaction_count"),
        F.max("timestamp").alias("last_interaction"),
        F.collect_set("interaction_type").alias("interaction_types")
    )
    
    print(f"✅ Created user-item matrix: {user_item_matrix.count()} user-item pairs")
    
    # Create user features for collaborative filtering
    print("🔄 Creating user behavior features...")
    
    user_features = interactions_df.groupBy("user_id").agg(
        F.count("*").alias("total_interactions"),
        F.countDistinct("product_id").alias("unique_products_interacted"),
        F.countDistinct("product_category").alias("unique_categories_interacted"),
        F.avg("product_price").alias("avg_price_interacted"),
        F.sum(F.when(F.col("interaction_type") == "purchase", 1).otherwise(0)).alias("total_purchases"),
        F.sum(F.when(F.col("interaction_type") == "view", 1).otherwise(0)).alias("total_views"),
        F.sum(F.when(F.col("interaction_type") == "like", 1).otherwise(0)).alias("total_likes"),
        F.max("timestamp").alias("last_activity"),
        F.min("timestamp").alias("first_activity")
    )
    
    # Calculate user activity level
    user_features = user_features.withColumn(
        "activity_level",
        F.when(F.col("total_interactions") >= 50, "high")
         .when(F.col("total_interactions") >= 20, "medium")
         .otherwise("low")
    )
    
    # Calculate purchase conversion rate
    user_features = user_features.withColumn(
        "purchase_conversion_rate",
        F.col("total_purchases") / F.col("total_interactions")
    )
    
    print(f"✅ Created user features: {user_features.count()} users")
    
    # Create item features for collaborative filtering
    print("🔄 Creating product popularity features...")
    
    item_features = interactions_df.groupBy("product_id").agg(
        F.count("*").alias("total_interactions"),
        F.countDistinct("user_id").alias("unique_users_interacted"),
        F.avg(F.when(F.col("rating").isNotNull(), F.col("rating"))).alias("avg_user_rating"),
        F.count(F.when(F.col("rating").isNotNull(), F.col("rating"))).alias("rating_count"),
        F.sum(F.when(F.col("interaction_type") == "purchase", 1).otherwise(0)).alias("total_purchases"),
        F.sum(F.when(F.col("interaction_type") == "view", 1).otherwise(0)).alias("total_views"),
        F.max("timestamp").alias("last_interaction")
    )
    
    # Calculate popularity score
    item_features = item_features.withColumn(
        "popularity_score",
        (F.col("total_interactions") * 0.3 + 
         F.col("unique_users_interacted") * 0.4 + 
         F.col("total_purchases") * 0.3)
    )
    
    # Calculate item conversion rate
    item_features = item_features.withColumn(
        "item_conversion_rate",
        F.col("total_purchases") / F.col("total_interactions")
    )
    
    print(f"✅ Created item features: {item_features.count()} products")
    
    return user_item_matrix, user_features, item_features

user_item_matrix, user_features, item_features = create_collaborative_filtering_features(interactions_df)

# ===== PART 4: FEATURE ENGINEERING FOR CONTENT-BASED FILTERING =====
print("\n=== 🏷️ Creating Content-Based Filtering Features ===")

def create_content_based_features(products_df, interactions_df):
    """Create product content features and user preference profiles"""
    
    print("🔄 Creating product content features...")
    
    # Product category one-hot encoding
    categories = products_df.select("category").distinct().rdd.flatMap(lambda x: x).collect()
    
    # Create category features
    product_content_features = products_df
    for category in categories:
        product_content_features = product_content_features.withColumn(
            f"category_{category.lower().replace(' ', '_')}",
            F.when(F.col("category") == category, 1.0).otherwise(0.0)
        )
    
    # Brand one-hot encoding
    brands = products_df.select("brand").distinct().rdd.flatMap(lambda x: x).collect()
    for brand in brands[:10]:  # Top 10 brands to avoid too many columns
        product_content_features = product_content_features.withColumn(
            f"brand_{brand.lower().replace(' ', '_')}",
            F.when(F.col("brand") == brand, 1.0).otherwise(0.0)
        )
    
    # Price range features
    product_content_features = product_content_features.withColumn(
        "price_range",
        F.when(F.col("price") < 100, "budget")
         .when(F.col("price") < 500, "mid_range")
         .when(F.col("price") < 1000, "premium")
         .otherwise("luxury")
    )
    
    # Normalize price for similarity calculations
    price_stats = products_df.agg(
        F.min("price").alias("min_price"),
        F.max("price").alias("max_price")
    ).collect()[0]
    
    product_content_features = product_content_features.withColumn(
        "price_normalized",
        (F.col("price") - price_stats["min_price"]) / 
        (price_stats["max_price"] - price_stats["min_price"])
    )
    
    # Rating features
    product_content_features = product_content_features.withColumn(
        "rating_normalized",
        F.col("rating") / 5.0
    )
    
    print(f"✅ Created content features for {product_content_features.count()} products")
    
    # Create user preference profiles
    print("🔄 Creating user preference profiles...")
    
    # User category preferences
    user_category_prefs = interactions_df.groupBy("user_id", "product_category").agg(
        F.count("*").alias("category_interactions")
    )
    
    # Calculate category preference percentages
    user_total_interactions = interactions_df.groupBy("user_id").agg(
        F.count("*").alias("total_interactions")
    )
    
    user_category_prefs = user_category_prefs.join(
        user_total_interactions, "user_id"
    ).withColumn(
        "category_preference_score",
        F.col("category_interactions") / F.col("total_interactions")
    )
    
    # Pivot to get user-category preference matrix
    user_content_preferences = user_category_prefs.groupBy("user_id").pivot("product_category").agg(
        F.first("category_preference_score")
    ).fillna(0.0)
    
    # User price preferences
    user_price_prefs = interactions_df.join(products_df, "product_id").groupBy("user_id").agg(
        F.avg("price").alias("avg_preferred_price"),
        F.min("price").alias("min_preferred_price"),
        F.max("price").alias("max_preferred_price"),
        F.stddev("price").alias("price_preference_std")
    )
    
    user_content_preferences = user_content_preferences.join(
        user_price_prefs, "user_id", "left"
    )
    
    print(f"✅ Created preference profiles for {user_content_preferences.count()} users")
    
    return product_content_features, user_content_preferences

product_content_features, user_content_preferences = create_content_based_features(products_df, interactions_df)

# ===== PART 5: CREATE TRAINING DATASETS =====
print("\n=== 🎯 Creating ML Training Datasets ===")

def create_training_datasets():
    """Create final datasets ready for ML model training"""
    
    print("🔄 Creating collaborative filtering training dataset...")
    
    # Collaborative filtering dataset: user-item pairs with ratings
    cf_training_data = user_item_matrix.select(
        "user_id",
        "product_id", 
        "interaction_score",
        "interaction_count"
    )
    
    # Add user and item features
    cf_training_data = cf_training_data.join(
        user_features.select("user_id", "activity_level", "purchase_conversion_rate"),
        "user_id",
        "left"
    ).join(
        item_features.select("product_id", "popularity_score", "item_conversion_rate"),
        "product_id",  
        "left"
    )
    
    print(f"✅ CF training dataset: {cf_training_data.count()} records")
    
    # Content-based filtering dataset: user-item pairs with content features
    print("🔄 Creating content-based filtering training dataset...")
    
    cb_training_data = interactions_df.select("user_id", "product_id").distinct()
    
    # Add product content features
    cb_training_data = cb_training_data.join(
        product_content_features.select(
            "product_id", "category", "brand", "price_normalized", 
            "rating_normalized", "price_range"
        ),
        "product_id",
        "left"
    )
    
    # Add user preferences
    cb_training_data = cb_training_data.join(
        user_content_preferences,
        "user_id",
        "left"
    )
    
    # Add target variable (interaction score from user-item matrix)
    cb_training_data = cb_training_data.join(
        user_item_matrix.select("user_id", "product_id", "interaction_score"),
        ["user_id", "product_id"],
        "left"
    ).fillna(0.0, ["interaction_score"])
    
    print(f"✅ CB training dataset: {cb_training_data.count()} records")
    
    return cf_training_data, cb_training_data

cf_training_data, cb_training_data = create_training_datasets()

# ===== PART 6: SAVE PROCESSED DATA =====
print("\n=== 💾 Saving Processed Data to S3 ===")

def save_processed_data():
    """Save all processed datasets to S3 in parquet format"""
    
    try:
        # Save collaborative filtering data
        print("💾 Saving collaborative filtering datasets...")
        
        user_item_matrix.coalesce(10).write.mode("overwrite").parquet(
            S3_PROCESSED_DATA_PATH + "collaborative_filtering/user_item_matrix"
        )
        
        user_features.coalesce(5).write.mode("overwrite").parquet(
            S3_PROCESSED_DATA_PATH + "collaborative_filtering/user_features"
        )
        
        item_features.coalesce(5).write.mode("overwrite").parquet(
            S3_PROCESSED_DATA_PATH + "collaborative_filtering/item_features"
        )
        
        cf_training_data.coalesce(10).write.mode("overwrite").parquet(
            S3_PROCESSED_DATA_PATH + "collaborative_filtering/training_data"
        )
        
        # Save content-based filtering data
        print("💾 Saving content-based filtering datasets...")
        
        product_content_features.coalesce(5).write.mode("overwrite").parquet(
            S3_PROCESSED_DATA_PATH + "content_based/product_features"
        )
        
        user_content_preferences.coalesce(5).write.mode("overwrite").parquet(
            S3_PROCESSED_DATA_PATH + "content_based/user_preferences"
        )
        
        cb_training_data.coalesce(10).write.mode("overwrite").parquet(
            S3_PROCESSED_DATA_PATH + "content_based/training_data"
        )
        
        # Save metadata
        print("💾 Saving processing metadata...")
        
        metadata = {
            "processing_timestamp": datetime.datetime.now().isoformat(),
            "data_counts": {
                "products": products_df.count(),
                "users": users_df.count(), 
                "interactions": interactions_df.count(),
                "user_item_pairs": user_item_matrix.count(),
                "cf_training_records": cf_training_data.count(),
                "cb_training_records": cb_training_data.count()
            },
            "data_quality_passed": data_quality_ok
        }
        
        # Convert to DataFrame and save
        metadata_df = spark.createDataFrame([metadata])
        metadata_df.coalesce(1).write.mode("overwrite").json(
            S3_PROCESSED_DATA_PATH + "metadata"
        )
        
        print("✅ All processed data saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False
    
    return True

save_success = save_processed_data()

# ===== SUMMARY AND CLEANUP =====
print(f"""
🎉 AWS GLUE ETL PIPELINE COMPLETE!

📊 Processing Summary:
✅ Raw data loaded: {products_df.count()} products, {users_df.count()} users, {interactions_df.count()} interactions
✅ Data quality validation: {'PASSED' if data_quality_ok else 'FAILED'}
✅ User-item matrix created: {user_item_matrix.count()} pairs
✅ Feature engineering completed for both CF and CB approaches
✅ Training datasets prepared and saved
✅ Data saved to S3: {'SUCCESS' if save_success else 'FAILED'}

🗂️ Processed Data Structure:
📁 collaborative_filtering/
  ├── user_item_matrix/ (interaction scores)
  ├── user_features/ (behavior patterns)
  ├── item_features/ (popularity metrics)
  └── training_data/ (ready for CF models)

📁 content_based/
  ├── product_features/ (content attributes)
  ├── user_preferences/ (preference profiles)
  └── training_data/ (ready for CB models)

📁 metadata/ (processing information)
""")

job.commit()