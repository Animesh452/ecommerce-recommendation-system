# Hybrid System - Clean Data Solution
# Load clean data and build hybrid system

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("🔥 Building Hybrid Recommendation System (Clean Data Solution)")
print(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# LOAD MODELS
print("\n=== 📥 Loading Clean Model Data ===")

def load_clean_models():
    """Load CF model and clean CB data"""
    
    try:
        # Load CF model (this works fine)
        with open('models/collaborative-filtering/cf_model.pkl', 'rb') as f:
            cf_artifacts = pickle.load(f)
        print(f"✅ CF model loaded: {len(cf_artifacts['user_to_idx']):,} users, {len(cf_artifacts['item_to_idx']):,} items")
        
        # Load clean CB data
        with open('models/content-based/cb_data_clean.pkl', 'rb') as f:
            cb_data = pickle.load(f)
        print(f"✅ CB data loaded: {len(cb_data['feature_columns'])} features, {len(cb_data['user_profiles_df']):,} user profiles")
        
        return cf_artifacts, cb_data
        
    except Exception as e:
        print(f"❌ Error loading clean models: {e}")
        return None, None

cf_artifacts, cb_data = load_clean_models()

if cf_artifacts is None or cb_data is None:
    print("❌ Run the CB data creation step first!")
    exit(1)

# Load original data
interactions_df = pd.read_csv("data/raw/interactions.csv")
products_df = pd.read_csv("data/raw/products.csv")
users_df = pd.read_csv("data/raw/users.csv")

# SIMPLE CONTENT-BASED RECOMMENDER
print("\n=== 🏗️ Building Simple Content-Based Recommender ===")

class SimpleContentRecommender:
    """Simple content-based recommender using clean data"""
    
    def __init__(self, product_features_df, user_profiles_df, feature_columns):
        self.product_features_df = product_features_df
        self.user_profiles_df = user_profiles_df
        self.feature_columns = feature_columns
        
        # Precompute product similarity
        self.product_matrix = product_features_df[feature_columns].fillna(0).values
        self.product_similarity = cosine_similarity(self.product_matrix)
        
        print(f"🏷️ Simple CB recommender ready with {len(feature_columns)} features")
    
    def get_user_recommendations(self, user_id, n_recommendations=10):
        """Get content-based recommendations"""
        
        # Check if user has profile
        user_profile = self.user_profiles_df[self.user_profiles_df['user_id'] == user_id]
        
        if user_profile.empty:
            # Cold start - return popular items
            top_products = self.product_features_df.nlargest(n_recommendations, 'rating')
            recommendations = []
            for _, product in top_products.iterrows():
                recommendations.append({
                    'product_id': product['product_id'],
                    'similarity_score': float(product['rating'] / 5.0),
                    'product_name': product['name'],
                    'category': product['category'],
                    'brand': product['brand'],
                    'price': float(product['price']),
                    'rating': float(product['rating'])
                })
            return recommendations
        
        # Use user profile for recommendations
        user_vector = user_profile[self.feature_columns].fillna(0).values[0]
        
        # Calculate similarity to all products
        scores = cosine_similarity([user_vector], self.product_matrix)[0]
        top_indices = np.argsort(scores)[::-1]
        
        recommendations = []
        for idx in top_indices:
            if len(recommendations) >= n_recommendations:
                break
                
            product = self.product_features_df.iloc[idx]
            recommendations.append({
                'product_id': product['product_id'],
                'similarity_score': float(scores[idx]),
                'product_name': product['name'],
                'category': product['category'],
                'brand': product['brand'],
                'price': float(product['price']),
                'rating': float(product['rating'])
            })
        
        return recommendations

# Create CB recommender
cb_recommender = SimpleContentRecommender(
    cb_data['product_features_df'], 
    cb_data['user_profiles_df'], 
    cb_data['feature_columns']
)

# HYBRID SYSTEM
print("\n=== 🔥 Building Hybrid System ===")

class SimpleHybridSystem:
    """Simple but effective hybrid recommendation system"""
    
    def __init__(self, cf_artifacts, cb_recommender, products_df):
        # CF components
        self.user_factors = cf_artifacts['user_factors']
        self.item_factors = cf_artifacts['item_factors']
        self.user_to_idx = cf_artifacts['user_to_idx']
        self.item_to_idx = cf_artifacts['item_to_idx']
        self.idx_to_user = cf_artifacts['idx_to_user']
        self.idx_to_item = cf_artifacts['idx_to_item']
        
        # CB component
        self.cb_recommender = cb_recommender
        self.products_df = products_df
        
        print(f"🔥 Simple hybrid system ready!")
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=10):
        """Get hybrid recommendations"""
        
        # Determine user type
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        interaction_count = len(user_interactions)
        
        if interaction_count < 5:
            user_type = 'cold_start'
            cf_weight, cb_weight = 0.2, 0.8
        elif interaction_count < 15:
            user_type = 'sparse'
            cf_weight, cb_weight = 0.4, 0.6
        elif interaction_count < 30:
            user_type = 'moderate' 
            cf_weight, cb_weight = 0.6, 0.4
        else:
            user_type = 'active'
            cf_weight, cb_weight = 0.7, 0.3
        
        print(f"🎯 User {user_id} ({user_type}): CF={cf_weight:.1f}, CB={cb_weight:.1f}")
        
        # Get CF recommendations
        cf_recs = []
        if user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            user_vector = self.user_factors[user_idx]
            item_scores = np.dot(user_vector, self.item_factors.T)
            top_indices = np.argsort(item_scores)[::-1]
            
            for item_idx in top_indices[:n_recommendations*2]:
                item_id = self.idx_to_item[item_idx]
                score = item_scores[item_idx]
                
                product_info = self.products_df[self.products_df['product_id'] == item_id]
                if not product_info.empty:
                    product = product_info.iloc[0]
                    cf_recs.append({
                        'product_id': item_id,
                        'cf_score': float(score),
                        'product_name': product['name'],
                        'category': product['category'],
                        'brand': product['brand'],
                        'price': float(product['price']),
                        'rating': float(product['rating'])
                    })
        
        # Get CB recommendations
        cb_recs_raw = self.cb_recommender.get_user_recommendations(user_id, n_recommendations*2)
        cb_recs = []
        for rec in cb_recs_raw:
            cb_recs.append({
                'product_id': rec['product_id'],
                'cb_score': rec['similarity_score'],
                'product_name': rec['product_name'],
                'category': rec['category'],
                'brand': rec['brand'],
                'price': rec['price'],
                'rating': rec['rating']
            })
        
        # Combine recommendations
        cf_dict = {rec['product_id']: rec for rec in cf_recs}
        cb_dict = {rec['product_id']: rec for rec in cb_recs}
        all_products = set(cf_dict.keys()) | set(cb_dict.keys())
        
        combined = []
        for product_id in all_products:
            cf_rec = cf_dict.get(product_id)
            cb_rec = cb_dict.get(product_id)
            
            cf_score = cf_rec['cf_score'] if cf_rec else 0
            cb_score = cb_rec['cb_score'] if cb_rec else 0
            
            # Simple normalization
            cf_norm = max(0, min(1, cf_score / 10))
            cb_norm = max(0, min(1, cb_score))
            
            hybrid_score = (cf_weight * cf_norm) + (cb_weight * cb_norm)
            
            product_info = cf_rec or cb_rec
            if product_info:
                combined.append({
                    'product_id': product_id,
                    'hybrid_score': float(hybrid_score),
                    'product_name': product_info['product_name'],
                    'category': product_info['category'],
                    'brand': product_info['brand'],
                    'price': product_info['price'],
                    'rating': product_info['rating'],
                    'source': f"{'CF' if cf_rec else ''}{'+ CB' if cb_rec else 'CB' if cb_rec else 'CF'}"
                })
        
        combined.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return combined[:n_recommendations]
    
    def compare_approaches(self, user_id, n_recommendations=5):
        """Compare all three approaches"""
        
        # CF recommendations
        cf_recs = []
        if user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            user_vector = self.user_factors[user_idx]
            item_scores = np.dot(user_vector, self.item_factors.T)
            top_indices = np.argsort(item_scores)[::-1]
            
            for item_idx in top_indices[:n_recommendations]:
                item_id = self.idx_to_item[item_idx]
                score = item_scores[item_idx]
                
                product_info = self.products_df[self.products_df['product_id'] == item_id]
                if not product_info.empty:
                    product = product_info.iloc[0]
                    cf_recs.append({
                        'product_name': product['name'],
                        'price': float(product['price']),
                        'cf_score': float(score)
                    })
        
        # CB recommendations
        cb_recs_raw = self.cb_recommender.get_user_recommendations(user_id, n_recommendations)
        cb_recs = [{
            'product_name': rec['product_name'],
            'price': rec['price'],
            'cb_score': rec['similarity_score']
        } for rec in cb_recs_raw]
        
        # Hybrid recommendations
        hybrid_recs_raw = self.get_hybrid_recommendations(user_id, n_recommendations)
        hybrid_recs = [{
            'product_name': rec['product_name'],
            'price': rec['price'],
            'hybrid_score': rec['hybrid_score'],
            'source': rec['source']
        } for rec in hybrid_recs_raw]
        
        return {
            'user_id': user_id,
            'collaborative': cf_recs,
            'content_based': cb_recs,
            'hybrid': hybrid_recs
        }

# Create hybrid system
hybrid_system = SimpleHybridSystem(cf_artifacts, cb_recommender, products_df)

# DEMONSTRATION
print("\n=== 🎯 Hybrid System Demonstration ===")

sample_users = users_df['user_id'].sample(3, random_state=42).tolist()

for i, user_id in enumerate(sample_users, 1):
    print(f"\n{'='*50}")
    print(f"👤 User {i}: {user_id}")
    
    # Show user interaction history
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    if not user_interactions.empty:
        categories = user_interactions.merge(products_df, on='product_id')['category'].value_counts()
        print(f"📊 Interactions: {dict(categories.head(3))}")
    
    # Compare all approaches
    comparison = hybrid_system.compare_approaches(user_id)
    
    print(f"\n🤝 Collaborative Filtering:")
    for j, rec in enumerate(comparison['collaborative'][:3], 1):
        print(f"   {j}. {rec['product_name']} (${rec['price']:.0f})")
    
    print(f"\n🏷️ Content-Based:")
    for j, rec in enumerate(comparison['content_based'][:3], 1):
        print(f"   {j}. {rec['product_name']} (${rec['price']:.0f})")
    
    print(f"\n🔥 Hybrid:")
    for j, rec in enumerate(comparison['hybrid'][:3], 1):
        print(f"   {j}. {rec['product_name']} (${rec['price']:.0f}) - {rec['source']}")

print(f"""
🎉 SIMPLE HYBRID SYSTEM COMPLETE!
""")