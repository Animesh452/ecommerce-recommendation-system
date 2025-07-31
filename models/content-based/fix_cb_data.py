# fix_cb_data.py
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os

print('🔧 Creating clean CB data...')

# Load data
interactions_df = pd.read_csv('data/raw/interactions.csv')
products_df = pd.read_csv('data/raw/products.csv')

# Create simple product features
products_features = products_df.copy()
category_dummies = pd.get_dummies(products_features['category'], prefix='category')

# Top brands
top_brands = products_features['brand'].value_counts().head(10).index
for brand in top_brands:
    products_features[f'brand_{brand.lower().replace(" ", "_")}'] = (products_features['brand'] == brand).astype(int)

# Normalize price and rating
scaler = StandardScaler()
products_features['price_normalized'] = scaler.fit_transform(products_features[['price']])
products_features['rating_normalized'] = products_features['rating'] / 5.0

# Combine features
brand_cols = [col for col in products_features.columns if col.startswith('brand_')]
all_features = pd.concat([
    products_features[['product_id', 'name', 'category', 'brand', 'price', 'rating']],
    category_dummies,
    products_features[brand_cols + ['price_normalized', 'rating_normalized']]
], axis=1)

feature_columns = [col for col in all_features.columns 
                  if col not in ['product_id', 'name', 'category', 'brand', 'price', 'rating']]

# Create simple user profiles
user_profiles_data = []
for user_id in interactions_df['user_id'].unique():
    user_profiles_data.append({'user_id': user_id})

user_profiles_df = pd.DataFrame(user_profiles_data)
for col in feature_columns:
    user_profiles_df[col] = np.random.random(len(user_profiles_df)) * 0.1

# Save clean data
os.makedirs('models/content-based', exist_ok=True)
cb_data = {
    'product_features_df': all_features,
    'user_profiles_df': user_profiles_df,
    'feature_columns': feature_columns,
    'training_date': datetime.now().isoformat()
}

with open('models/content-based/cb_data_clean.pkl', 'wb') as f:
    pickle.dump(cb_data, f)

print('✅ Clean CB data saved to cb_data_clean.pkl')
print(f'📊 Features: {len(feature_columns)}, Users: {len(user_profiles_df)}')