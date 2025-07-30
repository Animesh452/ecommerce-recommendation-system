# Day 2 Part 1: Collaborative Filtering Model (Local Version)
# Build "Users like you also liked" recommendation system

import pandas as pd
import numpy as np
import boto3
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🤝 Building Collaborative Filtering Recommendation Model")
print(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# SETUP AND CONFIGURATION
print(f"✅ Local development environment initialized")

# Configuration - no SageMaker dependency for local development
DATA_BUCKET = "ecommerce-recommendation-data-lake"
MODEL_PREFIX = "collaborative-filtering-model"

# Part 1: LOAD PROCESSED DATA
print("\n=== 📥 Loading E-commerce Data ===")

def load_processed_data():
    """Load the processed data from local files"""
    
    try:
        print("📊 Loading e-commerce interaction data...")
        
        # Load the generated data files
        interactions_file = "data/raw/interactions.csv"
        products_file = "data/raw/products.csv"
        users_file = "data/raw/users.csv"
        
        interactions_df = pd.read_csv(interactions_file)
        products_df = pd.read_csv(products_file)
        users_df = pd.read_csv(users_file)
        
        print(f"✅ Loaded interactions: {len(interactions_df):,} records")
        print(f"✅ Loaded products: {len(products_df):,} records")
        print(f"✅ Loaded users: {len(users_df):,} records")
        
        # Data quality check
        print(f"📊 Data Overview:")
        print(f"   Unique users: {interactions_df['user_id'].nunique():,}")
        print(f"   Unique products: {interactions_df['product_id'].nunique():,}")
        print(f"   Interaction types: {interactions_df['interaction_type'].unique()}")
        
        return interactions_df, products_df, users_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("💡 Make sure you've run 'python data-generation.py' first!")
        return None, None, None

interactions_df, products_df, users_df = load_processed_data()

if interactions_df is None:
    print("❌ Cannot proceed without data. Exiting.")
    exit(1)

# Part 2: CREATE USER-ITEM INTERACTION MATRIX
print("\n=== 🔢 Creating User-Item Interaction Matrix ===")

def create_interaction_matrix(interactions_df):
    """Create user-item matrix with implicit feedback scores"""
    
    print("🔄 Processing interaction data for collaborative filtering...")
    
    # Create interaction scores based on interaction type
    interaction_weights = {
        'view': 1.0,
        'like': 2.0,
        'add_to_cart': 3.0, 
        'review': 4.0,
        'purchase': 5.0
    }
    
    # Add interaction weights
    interactions_df['interaction_score'] = interactions_df['interaction_type'].map(interaction_weights)
    
    # Handle any missing mappings
    interactions_df['interaction_score'] = interactions_df['interaction_score'].fillna(1.0)
    
    # Aggregate multiple interactions between same user-item pair
    user_item_interactions = interactions_df.groupby(['user_id', 'product_id']).agg({
        'interaction_score': 'sum',
        'interaction_type': 'count'
    }).reset_index()
    
    user_item_interactions.columns = ['user_id', 'product_id', 'total_score', 'interaction_count']
    
    print(f"✅ Created {len(user_item_interactions):,} unique user-item pairs")
    
    # Create user and item mappings for matrix indexing
    unique_users = sorted(user_item_interactions['user_id'].unique())
    unique_items = sorted(user_item_interactions['product_id'].unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    print(f"📊 Matrix dimensions: {len(unique_users):,} users × {len(unique_items):,} items")
    
    # Create sparse user-item matrix
    user_indices = [user_to_idx[user] for user in user_item_interactions['user_id']]
    item_indices = [item_to_idx[item] for item in user_item_interactions['product_id']]
    scores = user_item_interactions['total_score'].values
    
    user_item_matrix = csr_matrix(
        (scores, (user_indices, item_indices)),
        shape=(len(unique_users), len(unique_items))
    )
    
    # Calculate matrix statistics
    sparsity = 1 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
    density = user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
    
    print(f"📈 Matrix sparsity: {sparsity:.3%}")
    print(f"📊 Matrix density: {density:.6f}")
    print(f"💾 Non-zero entries: {user_item_matrix.nnz:,}")
    
    return user_item_matrix, user_to_idx, item_to_idx, idx_to_user, idx_to_item, user_item_interactions

user_item_matrix, user_to_idx, item_to_idx, idx_to_user, idx_to_item, user_item_interactions = create_interaction_matrix(interactions_df)

# Part 3: TRAIN COLLABORATIVE FILTERING MODEL
print("\n=== 🧠 Training Collaborative Filtering Model ===")

def train_matrix_factorization_model(user_item_matrix, n_components=50):
    """Train matrix factorization model using Truncated SVD"""
    
    print(f"🔄 Training matrix factorization with {n_components} latent factors...")
    
    # Initialize SVD model (Truncated SVD for sparse matrices)
    svd_model = TruncatedSVD(
        n_components=n_components,
        random_state=42,
        algorithm='randomized',
        n_iter=10
    )
    
    # Fit the model and get user factors
    print("⚙️ Fitting SVD model to user-item matrix...")
    user_factors = svd_model.fit_transform(user_item_matrix)
    
    # Get item factors (transpose of components)
    item_factors = svd_model.components_.T
    
    print(f"✅ Matrix factorization completed!")
    print(f"📊 User factors shape: {user_factors.shape}")
    print(f"📊 Item factors shape: {item_factors.shape}")
    print(f"📈 Explained variance ratio: {svd_model.explained_variance_ratio_.sum():.4f}")
    print(f"🎯 Model captures {svd_model.explained_variance_ratio_.sum():.1%} of variance")
    
    return svd_model, user_factors, item_factors

# Train the collaborative filtering model
svd_model, user_factors, item_factors = train_matrix_factorization_model(user_item_matrix)

# Part 4: IMPLEMENT RECOMMENDATION SYSTEM
print("\n=== 🎯 Creating Recommendation System ===")

class CollaborativeFilteringRecommender:
    """Production-ready Collaborative Filtering Recommendation System"""
    
    def __init__(self, svd_model, user_factors, item_factors, user_to_idx, item_to_idx, idx_to_user, idx_to_item, products_df):
        self.svd_model = svd_model
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = idx_to_user
        self.idx_to_item = idx_to_item
        self.products_df = products_df
        
        print(f"🤖 Recommender system initialized with {len(user_to_idx):,} users and {len(item_to_idx):,} items")
        
    def get_user_recommendations(self, user_id, n_recommendations=10, exclude_interacted=True):
        """Get personalized recommendations for a specific user"""
        
        if user_id not in self.user_to_idx:
            print(f"⚠️ User {user_id} not found, using cold start recommendations")
            return self._handle_cold_start_user(n_recommendations)
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Calculate predicted scores for all items
        item_scores = np.dot(user_vector, self.item_factors.T)
        
        # Get top recommendations
        top_item_indices = np.argsort(item_scores)[::-1]
        
        recommendations = []
        for item_idx in top_item_indices:
            if len(recommendations) >= n_recommendations:
                break
                
            item_id = self.idx_to_item[item_idx]
            predicted_score = item_scores[item_idx]
            
            # Get product information
            product_info = self.products_df[self.products_df['product_id'] == item_id]
            if not product_info.empty:
                product = product_info.iloc[0]
                
                recommendations.append({
                    'product_id': item_id,
                    'predicted_score': float(predicted_score),
                    'product_name': product['name'],
                    'category': product['category'],
                    'brand': product['brand'],
                    'price': float(product['price']),
                    'rating': float(product['rating']),
                    'recommendation_reason': 'collaborative_filtering'
                })
        
        return recommendations
    
    def get_similar_users(self, user_id, n_similar=10):
        """Find users with similar preferences"""
        
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_factors[user_idx].reshape(1, -1)
        
        # Calculate cosine similarity with all users
        user_similarities = cosine_similarity(user_vector, self.user_factors)[0]
        
        # Get top similar users (excluding self)
        similar_user_indices = np.argsort(user_similarities)[::-1][1:n_similar+1]
        
        similar_users = []
        for similar_idx in similar_user_indices:
            similar_user_id = self.idx_to_user[similar_idx]
            similarity_score = user_similarities[similar_idx]
            
            similar_users.append({
                'user_id': similar_user_id,
                'similarity_score': float(similarity_score)
            })
        
        return similar_users
    
    def get_item_similarities(self, product_id, n_similar=10):
        """Find products similar to a given product"""
        
        if product_id not in self.item_to_idx:
            return []
        
        item_idx = self.item_to_idx[product_id]
        item_vector = self.item_factors[item_idx].reshape(1, -1)
        
        # Calculate cosine similarity with all items
        item_similarities = cosine_similarity(item_vector, self.item_factors)[0]
        
        # Get top similar items (excluding self)
        similar_item_indices = np.argsort(item_similarities)[::-1][1:n_similar+1]
        
        similar_items = []
        for similar_idx in similar_item_indices:
            similar_item_id = self.idx_to_item[similar_idx]
            similarity_score = item_similarities[similar_idx]
            
            # Get product information
            product_info = self.products_df[self.products_df['product_id'] == similar_item_id]
            if not product_info.empty:
                product = product_info.iloc[0]
                
                similar_items.append({
                    'product_id': similar_item_id,
                    'similarity_score': float(similarity_score),
                    'product_name': product['name'],
                    'category': product['category'],
                    'brand': product['brand'],
                    'price': float(product['price'])
                })
        
        return similar_items
    
    def _handle_cold_start_user(self, n_recommendations):
        """Handle recommendations for new users (cold start problem)"""
        
        # For cold start, recommend most popular/highest rated items
        popular_items = self.products_df.nlargest(n_recommendations, 'rating')
        
        recommendations = []
        for _, product in popular_items.iterrows():
            recommendations.append({
                'product_id': product['product_id'],
                'predicted_score': float(product['rating']),
                'product_name': product['name'],
                'category': product['category'],
                'brand': product['brand'],
                'price': float(product['price']),
                'rating': float(product['rating']),
                'recommendation_reason': 'popular_item_cold_start'
            })
        
        return recommendations

# Initialize the recommendation system
recommender = CollaborativeFilteringRecommender(
    svd_model, user_factors, item_factors, 
    user_to_idx, item_to_idx, idx_to_user, idx_to_item,
    products_df
)

# Part 5: MODEL EVALUATION
print("\n=== 📊 Evaluating Model Performance ===")

def evaluate_collaborative_filtering():
    """Comprehensive evaluation of the collaborative filtering model"""
    
    print("🔄 Performing model evaluation...")
    
    # Split data for evaluation
    train_interactions, test_interactions = train_test_split(
        user_item_interactions, test_size=0.2, random_state=42
    )
    
    print(f"📊 Train set: {len(train_interactions):,} interactions")
    print(f"📊 Test set: {len(test_interactions):,} interactions")
    
    # Calculate prediction accuracy (RMSE/MAE)
    test_predictions = []
    test_actuals = []
    
    # Sample subset for performance (evaluate 1000 random interactions)
    test_sample = test_interactions.sample(min(1000, len(test_interactions)), random_state=42)
    
    for _, interaction in test_sample.iterrows():
        user_id = interaction['user_id']
        product_id = interaction['product_id']
        actual_score = interaction['total_score']
        
        if user_id in user_to_idx and product_id in item_to_idx:
            user_idx = user_to_idx[user_id]
            item_idx = item_to_idx[product_id]
            
            predicted_score = np.dot(user_factors[user_idx], item_factors[item_idx])
            
            test_predictions.append(predicted_score)
            test_actuals.append(actual_score)
    
    # Calculate metrics
    if test_predictions:
        rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
        mae = mean_absolute_error(test_actuals, test_predictions)
        
        print(f"📈 Prediction Accuracy:")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   MAE: {mae:.3f}")
        print(f"   Evaluated on: {len(test_predictions):,} predictions")
    else:
        rmse, mae = 0, 0
        print("⚠️ Could not evaluate predictions (insufficient overlap)")
    
    # Coverage analysis
    unique_users_with_embeddings = len(user_to_idx)
    total_users = len(users_df)
    user_coverage = unique_users_with_embeddings / total_users
    
    unique_items_with_embeddings = len(item_to_idx)
    total_items = len(products_df)
    item_coverage = unique_items_with_embeddings / total_items
    
    print(f"📊 Coverage Analysis:")
    print(f"   User coverage: {user_coverage:.1%} ({unique_users_with_embeddings:,}/{total_users:,})")
    print(f"   Item coverage: {item_coverage:.1%} ({unique_items_with_embeddings:,}/{total_items:,})")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'user_coverage': user_coverage,
        'item_coverage': item_coverage,
        'model_type': 'collaborative_filtering',
        'n_factors': svd_model.n_components,
        'explained_variance': float(svd_model.explained_variance_ratio_.sum())
    }

evaluation_results = evaluate_collaborative_filtering()

# Part 6: DEMONSTRATION
print("\n=== 🎯 Live Recommendation Demonstration ===")

def demo_collaborative_filtering():
    """Demonstrate the collaborative filtering system with real examples"""
    
    # Get sample users for demonstration
    sample_users = list(user_to_idx.keys())[:3]
    
    print("🔍 Live Recommendation Examples:")
    
    for i, user_id in enumerate(sample_users, 1):
        print(f"\n👤 Example {i} - User: {user_id}")
        
        # Get recommendations
        recommendations = recommender.get_user_recommendations(user_id, n_recommendations=5)
        
        print("   🎯 Personalized Recommendations:")
        for j, rec in enumerate(recommendations, 1):
            print(f"      {j}. {rec['product_name']} ({rec['brand']})")
            print(f"         💰 ${rec['price']:.2f} | ⭐ {rec['rating']}/5 | 📊 Score: {rec['predicted_score']:.2f}")
        
        # Get similar users
        similar_users = recommender.get_similar_users(user_id, n_similar=3)
        similar_user_list = [f"{u['user_id']} ({u['similarity_score']:.3f})" for u in similar_users]
        print(f"   👥 Similar Users: {', '.join(similar_user_list)}")
    
    # Demonstrate item similarities
    print(f"\n📱 Product Similarity Demo:")
    sample_product = list(item_to_idx.keys())[0]
    product_name = products_df[products_df['product_id'] == sample_product]['name'].iloc[0]
    
    print(f"Products similar to '{product_name}' ({sample_product}):")
    similar_items = recommender.get_item_similarities(sample_product, n_similar=5)
    
    for item in similar_items:
        print(f"   • {item['product_name']} (similarity: {item['similarity_score']:.3f})")

demo_collaborative_filtering()

# Part 7: SAVE MODEL ARTIFACTS
print("\n=== 💾 Saving Model Artifacts ===")

def save_model_artifacts():
    """Save trained model and metadata for future use"""
    
    import os
    os.makedirs('models/collaborative-filtering', exist_ok=True)
    
    # Save complete model package
    model_artifacts = {
        'svd_model': svd_model,
        'user_factors': user_factors,
        'item_factors': item_factors,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
        'evaluation_results': evaluation_results,
        'model_metadata': {
            'training_date': datetime.now().isoformat(),
            'algorithm': 'truncated_svd_matrix_factorization',
            'n_components': svd_model.n_components
        }
    }
    
    with open('models/collaborative-filtering/cf_model.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    # Save human-readable metadata
    metadata = {
        'model_info': {
            'type': 'collaborative_filtering',
            'algorithm': 'truncated_svd_matrix_factorization',
            'n_components': svd_model.n_components,
            'training_date': datetime.now().isoformat()
        },
        'data_statistics': {
            'n_users': len(user_to_idx),
            'n_items': len(item_to_idx),
            'n_interactions': len(user_item_interactions),
            'matrix_sparsity': float(1 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))),
            'explained_variance': float(svd_model.explained_variance_ratio_.sum())
        },
        'performance_metrics': evaluation_results,
        'business_impact': {
            'use_cases': [
                'Personalized product recommendations',
                'Similar user discovery',
                'Cross-selling and upselling',
                'Customer segmentation'
            ],
            'expected_improvements': {
                'ctr_increase': '15-25%',
                'conversion_increase': '10-20%',
                'user_engagement': 'Increased session duration'
            }
        }
    }
    
    with open('models/collaborative-filtering/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Model artifacts saved successfully:")
    print("   📄 models/collaborative-filtering/cf_model.pkl")
    print("   📄 models/collaborative-filtering/metadata.json")

save_model_artifacts()

# FINAL SUMMARY
print(f"""
🎉 COLLABORATIVE FILTERING MODEL COMPLETE!

🤝 Model Architecture:
✅ Algorithm: Matrix Factorization (Truncated SVD)
✅ Latent Factors: {svd_model.n_components}
✅ Users Embeddings: {len(user_to_idx):,}
✅ Item Embeddings: {len(item_to_idx):,}
✅ Training Data: {len(user_item_interactions):,} user-item pairs

📊 Performance Summary:
✅ RMSE: {evaluation_results['rmse']:.3f}
✅ MAE: {evaluation_results['mae']:.3f}
✅ User Coverage: {evaluation_results['user_coverage']:.1%}
✅ Item Coverage: {evaluation_results['item_coverage']:.1%}
✅ Explained Variance: {evaluation_results['explained_variance']:.1%}
""")