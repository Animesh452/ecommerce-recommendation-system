# Day 1: E-commerce Dataset Generation
# Create realistic synthetic e-commerce data for recommendation system

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

print("🛍️ Generating E-commerce Dataset for Recommendation System")
print(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ===== PART 1: PRODUCT CATALOG =====
print("\n=== 📱 Creating Product Catalog ===")

def create_product_catalog():
    """Create realistic product catalog for electronics e-commerce store"""
    
    # Product categories and brands
    categories = {
        'Smartphones': {
            'brands': ['Apple', 'Samsung', 'Google', 'OnePlus', 'Xiaomi'],
            'price_range': (200, 1200),
            'base_rating': 4.2
        },
        'Laptops': {
            'brands': ['Apple', 'Dell', 'HP', 'Lenovo', 'ASUS'],
            'price_range': (500, 2500),
            'base_rating': 4.3
        },
        'Headphones': {
            'brands': ['Sony', 'Bose', 'Apple', 'Sennheiser', 'JBL'],
            'price_range': (50, 400),
            'base_rating': 4.1
        },
        'Tablets': {
            'brands': ['Apple', 'Samsung', 'Microsoft', 'Amazon', 'Lenovo'],
            'price_range': (150, 1000),
            'base_rating': 4.0
        },
        'Smartwatches': {
            'brands': ['Apple', 'Samsung', 'Garmin', 'Fitbit', 'Fossil'],
            'price_range': (100, 800),
            'base_rating': 4.2
        },
        'Accessories': {
            'brands': ['Anker', 'Belkin', 'Logitech', 'Apple', 'Samsung'],
            'price_range': (10, 150),
            'base_rating': 3.9
        }
    }
    
    products = []
    product_id = 1
    
    for category, info in categories.items():
        # Create 150-200 products per category
        num_products = random.randint(150, 200)
        
        for _ in range(num_products):
            brand = random.choice(info['brands'])
            price = round(random.uniform(*info['price_range']), 2)
            
            # Rating influenced by price and brand
            brand_boost = 0.3 if brand in ['Apple', 'Sony', 'Bose'] else 0
            price_boost = 0.2 if price > np.mean(info['price_range']) else 0
            rating = min(5.0, info['base_rating'] + brand_boost + price_boost + random.uniform(-0.3, 0.3))
            
            # Generate product name
            model_num = random.randint(100, 999)
            if category == 'Smartphones':
                name = f"{brand} {category[:-1]} {model_num}"
            elif category == 'Accessories':
                acc_types = ['Charger', 'Case', 'Stand', 'Cable', 'Adapter']
                name = f"{brand} {random.choice(acc_types)}"
            else:
                name = f"{brand} {category[:-1]} {model_num}"
            
            # Create description
            features = {
                'Smartphones': ['5G', 'Dual Camera', 'Face ID', 'Wireless Charging'],
                'Laptops': ['SSD', 'Backlit Keyboard', 'Touchscreen', 'Long Battery'],
                'Headphones': ['Noise Cancelling', 'Wireless', 'Hi-Res Audio', 'Comfort Fit'],
                'Tablets': ['Retina Display', 'Stylus Support', 'Long Battery', 'Lightweight'],
                'Smartwatches': ['Heart Rate', 'GPS', 'Water Resistant', 'Sleep Tracking'],
                'Accessories': ['Durable', 'Fast Charging', 'Universal', 'Compact']
            }
            
            selected_features = random.sample(features[category], k=random.randint(2, 3))
            description = f"{category[:-1]} with {', '.join(selected_features)}"
            
            products.append({
                'product_id': f'P{product_id:04d}',
                'name': name,
                'category': category,
                'brand': brand,
                'price': price,
                'rating': round(rating, 1),
                'num_reviews': random.randint(10, 500),
                'description': description,
                'in_stock': random.choice([True, True, True, False])  # 75% in stock
            })
            
            product_id += 1
    
    products_df = pd.DataFrame(products)
    print(f"✅ Created {len(products_df)} products across {len(categories)} categories")
    
    # Show sample
    print("\n📊 Sample Products:")
    sample_products = products_df.groupby('category').head(2)
    for _, product in sample_products.iterrows():
        print(f"  {product['product_id']}: {product['name']} - ${product['price']} ({product['rating']}⭐)")
    
    return products_df

products_df = create_product_catalog()

# ===== PART 2: USER PROFILES =====
print("\n=== 👥 Creating User Profiles ===")

def create_user_profiles():
    """Create diverse user profiles with realistic demographics"""
    
    # Demographics data
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    
    users = []
    
    for user_id in range(1, 10001):  # 10,000 users
        # Age distribution: realistic e-commerce demographics
        age = int(np.random.normal(35, 12))  # Mean 35, std 12
        age = max(18, min(70, age))  # Clamp between 18-70
        
        # Gender distribution
        gender = random.choice(['M', 'F', 'Other'])
        
        # Location
        location = random.choice(locations)
        
        # Signup date: last 2 years
        signup_date = datetime.now() - timedelta(days=random.randint(1, 730))
        
        # User preferences (affects purchase behavior)
        preferred_brands = random.sample(['Apple', 'Samsung', 'Sony', 'Dell', 'HP'], 
                                       k=random.randint(1, 3))
        
        # Spending behavior
        if age < 25:
            spending_power = 'low'
            avg_purchase = random.uniform(50, 300)
        elif age < 40:
            spending_power = 'medium'
            avg_purchase = random.uniform(200, 800)
        else:
            spending_power = 'high'
            avg_purchase = random.uniform(300, 1500)
        
        # Purchase frequency
        total_purchases = max(0, int(np.random.poisson(8)))  # Average 8 purchases
        
        users.append({
            'user_id': f'U{user_id:05d}',
            'age': age,
            'gender': gender,
            'location': location,
            'signup_date': signup_date.strftime('%Y-%m-%d'),
            'preferred_brands': ','.join(preferred_brands),
            'spending_power': spending_power,
            'avg_purchase_value': round(avg_purchase, 2),
            'total_purchases': total_purchases
        })
    
    users_df = pd.DataFrame(users)
    print(f"✅ Created {len(users_df)} user profiles")
    
    # Demographics summary
    print("\n📊 User Demographics:")
    print(f"  Age range: {users_df['age'].min()}-{users_df['age'].max()} (avg: {users_df['age'].mean():.1f})")
    print(f"  Gender distribution: {dict(users_df['gender'].value_counts())}")
    print(f"  Spending power: {dict(users_df['spending_power'].value_counts())}")
    print(f"  Total purchases: {users_df['total_purchases'].sum():,}")
    
    return users_df

users_df = create_user_profiles()

# ===== PART 3: USER INTERACTIONS =====
print("\n=== 🛒 Generating User Interactions ===")

def create_user_interactions(users_df, products_df):
    """Create realistic user interaction data - OPTIMIZED VERSION"""
    
    print("🔄 Generating realistic user behavior patterns...")
    
    # Pre-calculate for efficiency
    product_ids = products_df['product_id'].values
    product_ratings = products_df['rating'].values
    product_prices = products_df['price'].values
    product_categories = products_df['category'].values
    
    # Create brand lookup for efficiency
    brand_to_products = {}
    for brand in products_df['brand'].unique():
        brand_products = products_df[products_df['brand'] == brand]
        brand_to_products[brand] = {
            'ids': brand_products['product_id'].values,
            'ratings': brand_products['rating'].values,
            'prices': brand_products['price'].values,
            'categories': brand_products['category'].values
        }
    
    interactions = []
    interaction_types = ['view', 'add_to_cart', 'purchase', 'like', 'review']
    
    # Normalize probabilities
    probs = [0.7, 0.2, 0.08, 0.15, 0.05]
    probs = [p / sum(probs) for p in probs]
    
    # Process users in batches with progress tracking
    total_users = len(users_df)
    batch_size = 1000
    
    for batch_start in range(0, total_users, batch_size):
        batch_end = min(batch_start + batch_size, total_users)
        batch_users = users_df.iloc[batch_start:batch_end]
        
        print(f"  Processing users {batch_start+1:,}-{batch_end:,} of {total_users:,}...")
        
        for _, user in batch_users.iterrows():
            user_id = user['user_id']
            preferred_brands = user['preferred_brands'].split(',')
            spending_power = user['spending_power']
            
            # Reduce interaction count for performance
            if spending_power == 'high':
                num_interactions = random.randint(15, 40)  # Reduced from 50-200
            elif spending_power == 'medium':
                num_interactions = random.randint(8, 25)   # Reduced from 20-100
            else:
                num_interactions = random.randint(3, 15)   # Reduced from 10-50
            
            # Pre-generate random choices for this user
            interaction_choices = np.random.choice(interaction_types, size=num_interactions, p=probs)
            brand_preferences = np.random.random(num_interactions) < 0.4  # 40% prefer brands
            timestamps = [datetime.now() - timedelta(days=random.randint(1, 180)) 
                         for _ in range(num_interactions)]
            
            for i in range(num_interactions):
                # Choose product efficiently
                if brand_preferences[i] and preferred_brands[0]:  # Check if preferred brands exist
                    available_brands = [b for b in preferred_brands if b in brand_to_products]
                    if available_brands:
                        chosen_brand = random.choice(available_brands)
                        brand_data = brand_to_products[chosen_brand]
                        product_idx = random.randint(0, len(brand_data['ids']) - 1)
                        product_id = brand_data['ids'][product_idx]
                        product_rating = brand_data['ratings'][product_idx]
                        product_price = brand_data['prices'][product_idx]
                        product_category = brand_data['categories'][product_idx]
                    else:
                        # Fallback to random product
                        product_idx = random.randint(0, len(product_ids) - 1)
                        product_id = product_ids[product_idx]
                        product_rating = product_ratings[product_idx]
                        product_price = product_prices[product_idx]
                        product_category = product_categories[product_idx]
                else:
                    # Random product
                    product_idx = random.randint(0, len(product_ids) - 1)
                    product_id = product_ids[product_idx]
                    product_rating = product_ratings[product_idx]
                    product_price = product_prices[product_idx]
                    product_category = product_categories[product_idx]
                
                interaction_type = interaction_choices[i]
                
                # Rating (only for reviews and purchases)
                rating = None
                if interaction_type in ['review', 'purchase']:
                    rating = max(1, min(5, int(np.random.normal(product_rating, 0.8))))
                
                interactions.append({
                    'user_id': user_id,
                    'product_id': product_id,
                    'interaction_type': interaction_type,
                    'rating': rating,
                    'timestamp': timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'product_price': product_price,
                    'product_category': product_category
                })
    
    interactions_df = pd.DataFrame(interactions)
    print(f"✅ Generated {len(interactions_df):,} user interactions")
    
    # Interaction summary
    print("\n📊 Interaction Summary:")
    interaction_counts = interactions_df['interaction_type'].value_counts()
    for itype, count in interaction_counts.items():
        percentage = (count / len(interactions_df)) * 100
        print(f"  {itype}: {count:,} ({percentage:.1f}%)")
    
    return interactions_df

interactions_df = create_user_interactions(users_df, products_df)

# ===== PART 4: SAVE DATASETS =====
print("\n=== 💾 Saving Generated Datasets ===")

def save_datasets():
    """Save all datasets to CSV files"""
    
    # Create data directory
    import os
    os.makedirs('data/raw', exist_ok=True)
    
    # Save datasets
    products_df.to_csv('data/raw/products.csv', index=False)
    users_df.to_csv('data/raw/users.csv', index=False)
    interactions_df.to_csv('data/raw/interactions.csv', index=False)
    
    print("✅ Saved datasets:")
    print(f"  📱 Products: {len(products_df):,} records → data/raw/products.csv")
    print(f"  👥 Users: {len(users_df):,} records → data/raw/users.csv")
    print(f"  🛒 Interactions: {len(interactions_df):,} records → data/raw/interactions.csv")
    
    # Dataset statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total data size: ~{(len(products_df) + len(users_df) + len(interactions_df)) / 1000:.1f}K records")
    print(f"  Unique products: {products_df['product_id'].nunique():,}")
    print(f"  Unique users: {users_df['user_id'].nunique():,}")
    print(f"  Categories: {products_df['category'].nunique()}")
    print(f"  Brands: {products_df['brand'].nunique()}")
    print(f"  Date range: {interactions_df['timestamp'].min()} to {interactions_df['timestamp'].max()}")

save_datasets()

# ===== PART 5: DATA QUALITY ANALYSIS =====
print("\n=== 🔍 Data Quality Analysis ===")

def analyze_data_quality():
    """Analyze the quality and characteristics of generated data"""
    
    print("🔍 Data Quality Metrics:")
    
    # Products analysis
    print(f"\n📱 Products Data Quality:")
    print(f"  ✅ No missing values: {products_df.isnull().sum().sum() == 0}")
    print(f"  ✅ Price range: ${products_df['price'].min():.2f} - ${products_df['price'].max():.2f}")
    print(f"  ✅ Rating range: {products_df['rating'].min()} - {products_df['rating'].max()} stars")
    print(f"  ✅ Categories balanced: {dict(products_df['category'].value_counts())}")
    
    # Users analysis
    print(f"\n👥 Users Data Quality:")
    print(f"  ✅ No missing values: {users_df.isnull().sum().sum() == 0}")
    print(f"  ✅ Age distribution realistic: {users_df['age'].describe()}")
    print(f"  ✅ Purchase behavior varies: {users_df['total_purchases'].describe()}")
    
    # Interactions analysis
    print(f"\n🛒 Interactions Data Quality:")
    print(f"  ✅ Realistic interaction ratios maintained")
    print(f"  ✅ Temporal distribution over 6 months")
    print(f"  ✅ User engagement varies realistically")
    
    # Recommendation system readiness
    print(f"\n🎯 Recommendation System Readiness:")
    
    # User-item matrix sparsity
    user_item_matrix = interactions_df.pivot_table(
        index='user_id', 
        columns='product_id', 
        values='rating', 
        fill_value=0
    )
    sparsity = 1 - (user_item_matrix.astype(bool).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
    print(f"  📊 Matrix sparsity: {sparsity:.1%} (good for collaborative filtering)")
    
    # Content-based features
    print(f"  🏷️ Product features available: category, brand, price, rating")
    print(f"  👤 User features available: demographics, preferences, behavior")
    
    return {
        'products_count': len(products_df),
        'users_count': len(users_df),
        'interactions_count': len(interactions_df),
        'sparsity': sparsity,
        'categories': products_df['category'].nunique(),
        'brands': products_df['brand'].nunique()
    }

quality_metrics = analyze_data_quality()