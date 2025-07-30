# Day 2 Part 2: Content-Based Filtering Model
# Build "Products like this" recommendation system using product features

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🏷️ Building Content-Based Filtering Recommendation Model")
print(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Part 1: LOAD DATA
print("\n=== 📥 Loading E-commerce Data ===")

def load_data():
    """Load the e-commerce data for content-based analysis"""
    
    try:
        interactions_df = pd.read_csv("data/raw/interactions.csv")
        products_df = pd.read_csv("data/raw/products.csv")
        users_df = pd.read_csv("data/raw/users.csv")
        
        print(f"✅ Loaded interactions: {len(interactions_df):,} records")
        print(f"✅ Loaded products: {len(products_df):,} records")
        print(f"✅ Loaded users: {len(users_df):,} records")
        
        # Show product feature distribution
        print(f"\n📊 Product Feature Analysis:")
        print(f"   Categories: {products_df['category'].nunique()} ({list(products_df['category'].unique())})")
        print(f"   Brands: {products_df['brand'].nunique()} (top: {list(products_df['brand'].value_counts().head(5).index)})")
        print(f"   Price range: ${products_df['price'].min():.2f} - ${products_df['price'].max():.2f}")
        print(f"   Rating range: {products_df['rating'].min()} - {products_df['rating'].max()} stars")
        
        return interactions_df, products_df, users_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

interactions_df, products_df, users_df = load_data()

if interactions_df is None:
    print("❌ Cannot proceed without data. Exiting.")
    exit(1)

# Part 2: PRODUCT FEATURE ENGINEERING
print("\n=== 🔧 Engineering Product Content Features ===")

def create_product_features(products_df):
    """Create comprehensive product feature vectors for content-based filtering"""
    
    print("🔄 Creating product content features...")
    
    # Make a copy to avoid modifying original
    products_features = products_df.copy()
    
    # 1. Category Features (One-hot encoding)
    print("🏷️ Processing category features...")
    category_dummies = pd.get_dummies(products_features['category'], prefix='category')
    
    # 2. Brand Features (One-hot encoding for top brands)
    print("🏢 Processing brand features...")
    top_brands = products_features['brand'].value_counts().head(10).index
    for brand in top_brands:
        products_features[f'brand_{brand.lower().replace(" ", "_")}'] = (products_features['brand'] == brand).astype(int)
    
    # 3. Price Features (Multiple representations)
    print("💰 Processing price features...")
    
    # Normalize price
    scaler = StandardScaler()
    products_features['price_normalized'] = scaler.fit_transform(products_features[['price']])
    
    # Price categories
    price_percentiles = np.percentile(products_features['price'], [25, 50, 75])
    products_features['price_category'] = pd.cut(
        products_features['price'], 
        bins=[0, price_percentiles[0], price_percentiles[1], price_percentiles[2], float('inf')],
        labels=['budget', 'mid_range', 'premium', 'luxury']
    )
    
    # Price category dummies
    price_cat_dummies = pd.get_dummies(products_features['price_category'], prefix='price')
    
    # 4. Rating Features
    print("⭐ Processing rating features...")
    products_features['rating_normalized'] = products_features['rating'] / 5.0
    
    # Rating categories
    products_features['rating_category'] = pd.cut(
        products_features['rating'],
        bins=[0, 3.5, 4.0, 4.5, 5.0],
        labels=['low_rating', 'good_rating', 'high_rating', 'excellent_rating']
    )
    
    rating_cat_dummies = pd.get_dummies(products_features['rating_category'], prefix='rating')
    
    # 5. Text Features from Product Names and Descriptions
    print("📝 Processing text features...")
    
    # Combine name and description for text analysis
    products_features['text_content'] = products_features['name'] + ' ' + products_features['description']
    
    # TF-IDF on product text
    tfidf_vectorizer = TfidfVectorizer(
        max_features=100,  # Limit features for performance
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=2  # Ignore terms that appear in less than 2 documents
    )
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(products_features['text_content'])
    tfidf_feature_names = [f'text_{name}' for name in tfidf_vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=products_features.index)
    
    # 6. Combine All Features
    print("🔗 Combining all product features...")
    
    # Combine all feature sets
    all_features = pd.concat([
        products_features[['product_id', 'name', 'category', 'brand', 'price', 'rating']],  # Original info
        category_dummies,
        products_features[[col for col in products_features.columns if col.startswith('brand_')]],
        products_features[['price_normalized']],
        price_cat_dummies,
        products_features[['rating_normalized']],
        rating_cat_dummies,
        tfidf_df
    ], axis=1)
    
    # Get feature column names (excluding metadata)
    feature_columns = [col for col in all_features.columns 
                      if col not in ['product_id', 'name', 'category', 'brand', 'price', 'rating']]
    
    print(f"✅ Created {len(feature_columns)} product features:")
    print(f"   📂 Categories: {len(category_dummies.columns)}")
    print(f"   🏢 Brands: {len([col for col in feature_columns if col.startswith('brand_')])}")
    print(f"   💰 Price: {len([col for col in feature_columns if col.startswith('price')])}")
    print(f"   ⭐ Rating: {len([col for col in feature_columns if col.startswith('rating')])}")
    print(f"   📝 Text: {len(tfidf_feature_names)}")
    
    return all_features, feature_columns, tfidf_vectorizer

product_features_df, feature_columns, tfidf_vectorizer = create_product_features(products_df)

# Part 3: USER PREFERENCE PROFILING
print("\n=== 👤 Creating User Preference Profiles ===")

def create_user_preference_profiles(interactions_df, product_features_df, feature_columns):
    """Create user preference profiles based on interaction history"""
    
    print("🔄 Building user preference profiles from interaction history...")
    
    # Create interaction weights
    interaction_weights = {
        'view': 1.0,
        'like': 2.0,
        'add_to_cart': 3.0,
        'review': 4.0,
        'purchase': 5.0
    }
    
    # Add interaction weights to interactions
    interactions_weighted = interactions_df.copy()
    interactions_weighted['weight'] = interactions_weighted['interaction_type'].map(interaction_weights)
    
    # Merge with product features
    user_item_features = interactions_weighted.merge(
        product_features_df[['product_id'] + feature_columns], 
        on='product_id', 
        how='left'
    )
    
    print(f"📊 Merged interactions with features: {len(user_item_features):,} records")
    
    # Create weighted user preference profiles
    user_profiles = []
    
    for user_id in user_item_features['user_id'].unique():
        user_data = user_item_features[user_item_features['user_id'] == user_id]
        
        # Calculate weighted average of product features
        weights = user_data['weight'].values
        feature_matrix = user_data[feature_columns].values
        
        # Handle any NaN values
        feature_matrix = np.nan_to_num(feature_matrix, 0)
        
        # Weighted average (preference profile)
        if len(weights) > 0 and np.sum(weights) > 0:
            user_preference = np.average(feature_matrix, axis=0, weights=weights)
        else:
            user_preference = np.zeros(len(feature_columns))
        
        user_profiles.append({
            'user_id': user_id,
            'preference_vector': user_preference,
            'total_interactions': len(user_data),
            'total_weight': np.sum(weights)
        })
    
    # Convert to DataFrame
    user_profiles_df = pd.DataFrame(user_profiles)
    
    # Add preference vectors as separate columns
    preference_matrix = np.vstack(user_profiles_df['preference_vector'].values)
    preference_df = pd.DataFrame(preference_matrix, columns=feature_columns)
    user_profiles_df = pd.concat([
        user_profiles_df[['user_id', 'total_interactions', 'total_weight']], 
        preference_df
    ], axis=1)
    
    print(f"✅ Created preference profiles for {len(user_profiles_df):,} users")
    print(f"📊 Average interactions per user: {user_profiles_df['total_interactions'].mean():.1f}")
    
    return user_profiles_df

user_profiles_df = create_user_preference_profiles(interactions_df, product_features_df, feature_columns)

# Part 4: CONTENT-BASED RECOMMENDATION ENGINE
print("\n=== 🎯 Building Content-Based Recommendation Engine ===")

class ContentBasedRecommender:
    """Content-Based Filtering Recommendation System"""
    
    def __init__(self, product_features_df, user_profiles_df, feature_columns):
        self.product_features_df = product_features_df
        self.user_profiles_df = user_profiles_df
        self.feature_columns = feature_columns
        
        # Precompute product similarity matrix
        print("🔄 Precomputing product similarity matrix...")
        self.product_feature_matrix = product_features_df[feature_columns].values
        self.product_feature_matrix = np.nan_to_num(self.product_feature_matrix, 0)
        
        # Calculate product-product similarity
        self.product_similarity_matrix = cosine_similarity(self.product_feature_matrix)
        
        print(f"✅ Precomputed similarity for {len(product_features_df):,} products")
        
    def get_user_recommendations(self, user_id, n_recommendations=10, exclude_interacted=True):
        """Get content-based recommendations for a user"""
        
        # Get user preference profile
        user_profile = self.user_profiles_df[self.user_profiles_df['user_id'] == user_id]
        
        if user_profile.empty:
            return self._handle_cold_start_user(n_recommendations)
        
        # Get user preference vector
        user_preference_vector = user_profile[self.feature_columns].values[0]
        user_preference_vector = np.nan_to_num(user_preference_vector, 0)
        
        # Calculate similarity between user preferences and all products
        product_scores = cosine_similarity([user_preference_vector], self.product_feature_matrix)[0]
        
        # Get top recommendations
        top_product_indices = np.argsort(product_scores)[::-1]
        
        recommendations = []
        for product_idx in top_product_indices:
            if len(recommendations) >= n_recommendations:
                break
            
            product_row = self.product_features_df.iloc[product_idx]
            similarity_score = product_scores[product_idx]
            
            recommendations.append({
                'product_id': product_row['product_id'],
                'similarity_score': float(similarity_score),
                'product_name': product_row['name'],
                'category': product_row['category'],
                'brand': product_row['brand'],
                'price': float(product_row['price']),
                'rating': float(product_row['rating']),
                'recommendation_reason': 'content_based_user_profile'
            })
        
        return recommendations
    
    def get_similar_products(self, product_id, n_similar=10):
        """Find products similar to a given product based on content features"""
        
        # Find product index
        product_row = self.product_features_df[self.product_features_df['product_id'] == product_id]
        
        if product_row.empty:
            return []
        
        product_idx = product_row.index[0]
        
        # Get similarity scores for this product
        similarity_scores = self.product_similarity_matrix[product_idx]
        
        # Get top similar products (excluding self)
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_similar+1]
        
        similar_products = []
        for similar_idx in similar_indices:
            similar_product = self.product_features_df.iloc[similar_idx]
            similarity_score = similarity_scores[similar_idx]
            
            similar_products.append({
                'product_id': similar_product['product_id'],
                'similarity_score': float(similarity_score),
                'product_name': similar_product['name'],
                'category': similar_product['category'],
                'brand': similar_product['brand'],
                'price': float(similar_product['price']),
                'rating': float(similar_product['rating'])
            })
        
        return similar_products
    
    def get_category_recommendations(self, category, user_id=None, n_recommendations=10):
        """Get recommendations within a specific category"""
        
        category_products = self.product_features_df[self.product_features_df['category'] == category]
        
        if user_id and not self.user_profiles_df[self.user_profiles_df['user_id'] == user_id].empty:
            # Personalized category recommendations
            user_preference = self.user_profiles_df[self.user_profiles_df['user_id'] == user_id][self.feature_columns].values[0]
            user_preference = np.nan_to_num(user_preference, 0)
            
            category_feature_matrix = category_products[self.feature_columns].values
            category_feature_matrix = np.nan_to_num(category_feature_matrix, 0)
            
            scores = cosine_similarity([user_preference], category_feature_matrix)[0]
            top_indices = np.argsort(scores)[::-1][:n_recommendations]
            
            recommendations = []
            for idx in top_indices:
                product = category_products.iloc[idx]
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
        else:
            # Generic category recommendations (highest rated)
            top_in_category = category_products.nlargest(n_recommendations, 'rating')
            
            recommendations = []
            for _, product in top_in_category.iterrows():
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
    
    def _handle_cold_start_user(self, n_recommendations):
        """Handle recommendations for new users without interaction history"""
        
        # Recommend highest-rated products across categories
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
                'rating': float(product['rating']),
                'recommendation_reason': 'cold_start_highest_rated'
            })
        
        return recommendations

# Initialize the content-based recommender
cb_recommender = ContentBasedRecommender(product_features_df, user_profiles_df, feature_columns)

# Part 5: MODEL EVALUATION
print("\n=== 📊 Evaluating Content-Based Model ===")

def evaluate_content_based_model():
    """Evaluate the content-based recommendation model"""
    
    print("🔄 Evaluating content-based recommendation performance...")
    
    # Test on sample of users
    sample_users = user_profiles_df['user_id'].sample(min(100, len(user_profiles_df)), random_state=42)
    
    evaluation_metrics = {
        'recommendation_coverage': 0,
        'category_diversity': 0,
        'avg_similarity_score': 0,
        'cold_start_handling': 0
    }
    
    total_recommendations = 0
    total_similarity = 0
    categories_recommended = set()
    
    for user_id in sample_users:
        recommendations = cb_recommender.get_user_recommendations(user_id, n_recommendations=10)
        
        if recommendations:
            total_recommendations += len(recommendations)
            total_similarity += sum([rec['similarity_score'] for rec in recommendations])
            categories_recommended.update([rec['category'] for rec in recommendations])
    
    # Calculate metrics
    if total_recommendations > 0:
        evaluation_metrics['recommendation_coverage'] = len(sample_users) / len(user_profiles_df)
        evaluation_metrics['avg_similarity_score'] = total_similarity / total_recommendations
        evaluation_metrics['category_diversity'] = len(categories_recommended) / product_features_df['category'].nunique()
    
    # Test cold start handling
    fake_user_recs = cb_recommender.get_user_recommendations('FAKE_USER_123', n_recommendations=5)
    evaluation_metrics['cold_start_handling'] = len(fake_user_recs) > 0
    
    print(f"📊 Content-Based Model Evaluation:")
    print(f"   Average similarity score: {evaluation_metrics['avg_similarity_score']:.3f}")
    print(f"   Category diversity: {evaluation_metrics['category_diversity']:.1%}")
    print(f"   Cold start handling: {'✅ Yes' if evaluation_metrics['cold_start_handling'] else '❌ No'}")
    print(f"   Recommendation coverage: {evaluation_metrics['recommendation_coverage']:.1%}")
    
    return evaluation_metrics

cb_evaluation = evaluate_content_based_model()

# Part 6: DEMONSTRATION
print("\n=== 🎯 Content-Based Recommendation Demonstration ===")

def demo_content_based_filtering():
    """Demonstrate the content-based filtering system"""
    
    # Get sample users for demonstration
    sample_users = user_profiles_df['user_id'].head(3).tolist()
    
    print("🔍 Content-Based Recommendation Examples:")
    
    for i, user_id in enumerate(sample_users, 1):
        print(f"\n👤 Example {i} - User: {user_id}")
        
        # Get user's interaction summary
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        user_categories = user_interactions.merge(products_df, on='product_id')['category'].value_counts()
        
        print(f"   📊 User's interaction history: {user_categories.to_dict()}")
        
        # Get content-based recommendations
        recommendations = cb_recommender.get_user_recommendations(user_id, n_recommendations=5)
        
        print("   🎯 Content-Based Recommendations:")
        for j, rec in enumerate(recommendations, 1):
            print(f"      {j}. {rec['product_name']} ({rec['brand']})")
            print(f"         💰 ${rec['price']:.2f} | ⭐ {rec['rating']}/5 | 📊 Score: {rec['similarity_score']:.3f}")
    
    # Demonstrate product similarity
    print(f"\n📱 Product Similarity Demo (Content-Based):")
    sample_product = products_df.iloc[0]
    print(f"Products similar to '{sample_product['name']}' ({sample_product['category']}):")
    
    similar_products = cb_recommender.get_similar_products(sample_product['product_id'], n_similar=5)
    
    for product in similar_products:
        print(f"   • {product['product_name']} (similarity: {product['similarity_score']:.3f})")
        print(f"     {product['category']} | {product['brand']} | ${product['price']:.2f}")
    
    # Demonstrate category recommendations
    print(f"\n📂 Category-Based Recommendations:")
    sample_category = products_df['category'].value_counts().index[0]
    category_recs = cb_recommender.get_category_recommendations(sample_category, n_recommendations=3)
    
    print(f"Top {sample_category} recommendations:")
    for rec in category_recs:
        print(f"   • {rec['product_name']} (score: {rec['similarity_score']:.3f})")

demo_content_based_filtering()

# Part 7: SAVE MODEL ARTIFACTS
print("\n=== 💾 Saving Content-Based Model ===")

def save_content_based_model():
    """Save the content-based model artifacts"""
    
    import os
    os.makedirs('models/content-based', exist_ok=True)
    
    # Save model components
    model_artifacts = {
        'recommender': cb_recommender,
        'product_features_df': product_features_df,
        'user_profiles_df': user_profiles_df,
        'feature_columns': feature_columns,
        'tfidf_vectorizer': tfidf_vectorizer,
        'evaluation_results': cb_evaluation
    }
    
    with open('models/content-based/cb_model.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    # Save metadata
    metadata = {
        'model_info': {
            'type': 'content_based_filtering',
            'algorithm': 'cosine_similarity_feature_matching',
            'features_used': len(feature_columns),
            'training_date': datetime.now().isoformat()
        },
        'data_statistics': {
            'n_products': len(product_features_df),
            'n_users_with_profiles': len(user_profiles_df),
            'n_features': len(feature_columns),
            'feature_breakdown': {
                'categories': len([col for col in feature_columns if col.startswith('category_')]),
                'brands': len([col for col in feature_columns if col.startswith('brand_')]),
                'price_features': len([col for col in feature_columns if col.startswith('price')]),
                'rating_features': len([col for col in feature_columns if col.startswith('rating')]),
                'text_features': len([col for col in feature_columns if col.startswith('text_')])
            }
        },
        'performance_metrics': cb_evaluation,
        'business_applications': {
            'use_cases': [
                'Product similarity recommendations',
                'User preference-based suggestions', 
                'Category-specific recommendations',
                'Cold start user handling',
                'Content discovery and exploration'
            ],
            'advantages': [
                'No cold start problem for products',
                'Explainable recommendations',
                'Diverse recommendations across categories',
                'Works for new users immediately'
            ]
        }
    }
    
    with open('models/content-based/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Content-based model saved successfully:")
    print("   📄 models/content-based/cb_model.pkl")
    print("   📄 models/content-based/metadata.json")

save_content_based_model()

# FINAL SUMMARY
print(f"""
🎉 CONTENT-BASED FILTERING MODEL COMPLETE!

🏷️ Model Architecture:
✅ Feature Engineering: {len(feature_columns)} product features
✅ User Profiles: {len(user_profiles_df):,} preference vectors
✅ Product Similarity: Precomputed {len(product_features_df):,}×{len(product_features_df):,} matrix
✅ Algorithm: Cosine similarity on feature vectors

📊 Feature Breakdown:
✅ Categories: {len([col for col in feature_columns if col.startswith('category_')])} one-hot encoded
✅ Brands: {len([col for col in feature_columns if col.startswith('brand_')])} top brands encoded
✅ Price: {len([col for col in feature_columns if col.startswith('price')])} normalized & categorized
✅ Ratings: {len([col for col in feature_columns if col.startswith('rating')])} normalized & categorized  
✅ Text: {len([col for col in feature_columns if col.startswith('text_')])} TF-IDF features
""")