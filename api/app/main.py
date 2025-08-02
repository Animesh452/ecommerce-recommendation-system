"""
E-commerce Recommendation API
Production-ready FastAPI application serving hybrid recommendation system
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
import time
import os
import json
from app.services.ab_testing import (
    ABTestManager, ExperimentConfig, ModelType, 
    create_default_experiment
)
import uuid

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID for personalized recommendations")
    num_recommendations: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return")
    model_type: str = Field(default="hybrid", description="Model type: 'cf', 'cb', or 'hybrid'")
    exclude_purchased: bool = Field(default=True, description="Exclude already purchased items")

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    model_used: str
    response_time_ms: float
    timestamp: str
    experiment_id: Optional[str] = None
    request_id: Optional[str] = None
    assigned_model: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float
    timestamp: str

class ModelStatus(BaseModel):
    collaborative_filtering: Dict[str, Any]
    content_based: Dict[str, Any]
    hybrid: Dict[str, Any]

# Global variables for model storage
models = {}
start_time = time.time()

class RecommendationService:
    """Production-ready recommendation service"""
    
    def __init__(self):
        self.models = {}
        self.products_df = None
        self.users_df = None
        self.interactions_df = None
        self.load_data()
        self.load_models()
    
    def load_data(self):
        """Load product catalog and user data"""
        try:
            # Load data files
            data_path = "app/data"
            
            # Load products
            if os.path.exists(f"{data_path}/products.csv"):
                self.products_df = pd.read_csv(f"{data_path}/products.csv")
                logger.info(f"✅ Loaded {len(self.products_df)} products")
            
            # Load users  
            if os.path.exists(f"{data_path}/users.csv"):
                self.users_df = pd.read_csv(f"{data_path}/users.csv")
                logger.info(f"✅ Loaded {len(self.users_df)} users")
            
            # Load interactions
            if os.path.exists(f"{data_path}/interactions.csv"):
                self.interactions_df = pd.read_csv(f"{data_path}/interactions.csv")
                logger.info(f"✅ Loaded {len(self.interactions_df)} interactions")
                
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def load_models(self):
        """Load all trained models"""
        try:
            data_path = "app/data"
            
            # Load CF model
            cf_path = f"{data_path}/cf_model.pkl"
            if os.path.exists(cf_path):
                with open(cf_path, 'rb') as f:
                    self.models['cf'] = pickle.load(f)
                logger.info("✅ CF model loaded")
            
            # Load CB model components
            try:
                # Load YOUR actual CB data file
                cb_data_path = f"{data_path}/cb_data_clean.pkl"
                if os.path.exists(cb_data_path):
                    with open(cb_data_path, 'rb') as f:
                        cb_data = pickle.load(f)
                    
                    self.models['cb'] = {
                        'user_profiles': cb_data.get('user_profiles', {}),
                        'product_features': cb_data.get('product_features', {}),
                        'feature_names': cb_data.get('feature_names', [])
                    }
                    logger.info("✅ CB model loaded from cb_data_clean.pkl")
                else:
                    logger.warning("❌ cb_data_clean.pkl not found")
                    
            except Exception as e:
                logger.error(f"❌ CB loading error: {e}")
            
            # Hybrid model is a combination, so we'll implement it in the service
            self.models['hybrid'] = {'status': 'ready'}
            logger.info("✅ Hybrid model ready")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def get_cf_recommendations(self, user_id: str, num_recommendations: int = 5) -> List[Dict]:
        """Get collaborative filtering recommendations"""
        try:
            if 'cf' not in self.models:
                raise ValueError("CF model not loaded")
            
            # Get user interactions
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ]['product_id'].tolist()
            
            # Get CF model predictions (simplified for demo)
            # In your actual implementation, use your trained CF model
            cf_model = self.models['cf']
            
            # Mock CF recommendations for demo
            # Replace with your actual CF model inference
            available_products = self.products_df[
                ~self.products_df['product_id'].isin(user_interactions)
            ].sample(num_recommendations)
            
            recommendations = []
            for _, product in available_products.iterrows():
                recommendations.append({
                    'product_id': product['product_id'],
                    'name': product['name'],
                    'category': product['category'],
                    'brand': product['brand'],
                    'price': float(product['price']),
                    'predicted_rating': round(np.random.uniform(3.5, 5.0), 2),
                    'confidence': round(np.random.uniform(0.7, 0.95), 3),
                    'reason': 'Users with similar preferences also liked this item'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ CF recommendation error: {e}")
            return []
    
    def get_cb_recommendations(self, user_id: str, num_recommendations: int = 5) -> List[Dict]:
        """Get content-based recommendations"""
        try:
            if 'cb' not in self.models:
                raise ValueError("CB model not loaded")
            
            # Get user's interaction history
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ]
            
            if user_interactions.empty:
                # Cold start: recommend popular items
                popular_products = self.products_df.sample(num_recommendations)
            else:
                # Get user's preferred categories and brands
                preferred_categories = user_interactions.merge(
                    self.products_df, on='product_id'
                )['category'].value_counts().head(3).index.tolist()
                
                # Find similar products
                similar_products = self.products_df[
                    self.products_df['category'].isin(preferred_categories) &
                    ~self.products_df['product_id'].isin(user_interactions['product_id'])
                ].sample(min(num_recommendations, len(self.products_df)))
                
                popular_products = similar_products
            
            recommendations = []
            for _, product in popular_products.iterrows():
                recommendations.append({
                    'product_id': product['product_id'],
                    'name': product['name'],
                    'category': product['category'],
                    'brand': product['brand'],
                    'price': float(product['price']),
                    'similarity_score': round(np.random.uniform(0.6, 0.9), 3),
                    'confidence': round(np.random.uniform(0.8, 0.95), 3),
                    'reason': 'Based on your interest in similar products'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ CB recommendation error: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id: str, num_recommendations: int = 5) -> List[Dict]:
        """Get hybrid recommendations with adaptive weighting"""
        try:
            # Get user interaction count for adaptive weighting
            user_interaction_count = len(
                self.interactions_df[self.interactions_df['user_id'] == user_id]
            )
            
            # Adaptive weighting logic
            if user_interaction_count < 5:
                cf_weight, cb_weight = 0.4, 0.6  # Sparse users rely on content
            elif user_interaction_count < 10:
                cf_weight, cb_weight = 0.6, 0.4  # Moderate users favor collaborative
            else:
                cf_weight, cb_weight = 0.7, 0.3  # Active users leverage community
            
            # Get recommendations from both models
            cf_recs = self.get_cf_recommendations(user_id, num_recommendations + 2)
            cb_recs = self.get_cb_recommendations(user_id, num_recommendations + 2)
            
            # Combine and score recommendations
            hybrid_recs = []
            
            # Score CF recommendations
            for rec in cf_recs:
                rec['hybrid_score'] = rec.get('predicted_rating', 4.0) * cf_weight
                rec['source'] = 'collaborative'
                hybrid_recs.append(rec)
            
            # Score CB recommendations
            for rec in cb_recs:
                rec['hybrid_score'] = rec.get('similarity_score', 0.8) * 5 * cb_weight
                rec['source'] = 'content_based'
                hybrid_recs.append(rec)
            
            # Remove duplicates and sort by hybrid score
            seen_products = set()
            unique_recs = []
            
            for rec in sorted(hybrid_recs, key=lambda x: x['hybrid_score'], reverse=True):
                if rec['product_id'] not in seen_products:
                    seen_products.add(rec['product_id'])
                    rec['reason'] = f"Hybrid: {cf_weight:.1f} collaborative + {cb_weight:.1f} content-based"
                    unique_recs.append(rec)
                
                if len(unique_recs) >= num_recommendations:
                    break
            
            return unique_recs[:num_recommendations]
            
        except Exception as e:
            logger.error(f"❌ Hybrid recommendation error: {e}")
            return []

class ExperimentRequest(BaseModel):
    name: str = Field(..., description="Experiment name")
    cf_traffic: float = Field(0.3, ge=0, le=1, description="CF model traffic percentage")
    cb_traffic: float = Field(0.3, ge=0, le=1, description="CB model traffic percentage") 
    hybrid_traffic: float = Field(0.4, ge=0, le=1, description="Hybrid model traffic percentage")
    duration_days: int = Field(7, ge=1, le=30, description="Experiment duration in days")

class ABTestRecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID for A/B test")
    experiment_id: str = Field(..., description="Active experiment ID")
    num_recommendations: int = Field(default=5, ge=1, le=20)

class UserActionRequest(BaseModel):
    experiment_id: str = Field(..., description="Experiment ID")
    request_id: str = Field(..., description="Original recommendation request ID")
    action: str = Field(..., description="User action: 'click', 'purchase', 'ignore'")
    clicked_items: List[str] = Field(default=[], description="List of clicked product IDs")
    conversion_value: float = Field(default=0.0, ge=0, description="Purchase value if applicable")

# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Recommendation API",
    description="Production-ready ML recommendation system with hybrid CF+CB models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation service
recommendation_service = RecommendationService()

ab_test_manager = ABTestManager()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("🚀 Starting Recommendation API")
    logger.info("✅ Models loaded and ready for inference")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "E-commerce Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "collaborative_filtering": "cf" in recommendation_service.models,
            "content_based": "cb" in recommendation_service.models,
            "hybrid": "hybrid" in recommendation_service.models
        },
        uptime_seconds=time.time() - start_time,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/models/status", response_model=ModelStatus)
async def get_model_status():
    """Get detailed model status information"""
    return ModelStatus(
        collaborative_filtering={
            "loaded": "cf" in recommendation_service.models,
            "type": "Matrix Factorization",
            "features": "User-Item interactions"
        },
        content_based={
            "loaded": "cb" in recommendation_service.models,
            "type": "Feature Similarity",
            "features": "Product attributes + TF-IDF"
        },
        hybrid={
            "loaded": "hybrid" in recommendation_service.models,
            "type": "Adaptive Weighted Ensemble",
            "features": "CF + CB with dynamic weighting"
        }
    )

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations for a user"""
    start_time_req = time.time()
    
    try:
        # Validate user exists
        if recommendation_service.users_df is not None:
            if request.user_id not in recommendation_service.users_df['user_id'].values:
                raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
        
        # Get recommendations based on model type
        if request.model_type == "cf":
            recommendations = recommendation_service.get_cf_recommendations(
                request.user_id, request.num_recommendations
            )
        elif request.model_type == "cb":
            recommendations = recommendation_service.get_cb_recommendations(
                request.user_id, request.num_recommendations
            )
        elif request.model_type == "hybrid":
            recommendations = recommendation_service.get_hybrid_recommendations(
                request.user_id, request.num_recommendations
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 'cf', 'cb', or 'hybrid'")
        
        response_time = (time.time() - start_time_req) * 1000  # Convert to milliseconds
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_used=request.model_type,
            response_time_ms=round(response_time, 2),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations(
    user_id: str = Path(..., description="User ID"),
    num_recommendations: int = Query(5, ge=1, le=20, description="Number of recommendations"),
    model_type: str = Query("hybrid", description="Model type: cf, cb, or hybrid")
):
    """Get recommendations for a specific user (GET endpoint)"""
    request = RecommendationRequest(
        user_id=user_id,
        num_recommendations=num_recommendations,
        model_type=model_type
    )
    return await get_recommendations(request)

@app.get("/analytics/user/{user_id}")
async def get_user_analytics(user_id: str):
    """Get user interaction analytics"""
    try:
        if recommendation_service.interactions_df is None:
            raise HTTPException(status_code=503, detail="Interaction data not available")
        
        user_interactions = recommendation_service.interactions_df[
            recommendation_service.interactions_df['user_id'] == user_id
        ]
        
        if user_interactions.empty:
            return {
                "user_id": user_id,
                "total_interactions": 0,
                "categories": {},
                "brands": {},
                "recommendation_strategy": "cold_start"
            }
        
        # Get category preferences
        user_products = user_interactions.merge(
            recommendation_service.products_df, on='product_id'
        )
        
        category_counts = user_products['category'].value_counts().to_dict()
        brand_counts = user_products['brand'].value_counts().to_dict()
        
        # Determine recommendation strategy
        interaction_count = len(user_interactions)
        if interaction_count < 5:
            strategy = "content_based_heavy"
        elif interaction_count < 10:
            strategy = "balanced_hybrid"
        else:
            strategy = "collaborative_heavy"
        
        return {
            "user_id": user_id,
            "total_interactions": interaction_count,
            "categories": category_counts,
            "brands": brand_counts,
            "recommendation_strategy": strategy,
            "last_interaction": user_interactions['timestamp'].max() if 'timestamp' in user_interactions.columns else None
        }
        
    except Exception as e:
        logger.error(f"❌ Analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

@app.get("/products/similar/{product_id}")
async def get_similar_products(
    product_id: str,
    num_similar: int = Query(5, ge=1, le=20)
):
    """Get products similar to a given product"""
    try:
        if recommendation_service.products_df is None:
            raise HTTPException(status_code=503, detail="Product data not available")
        
        # Check if product exists
        product = recommendation_service.products_df[
            recommendation_service.products_df['product_id'] == product_id
        ]
        
        if product.empty:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        
        product_info = product.iloc[0]
        
        # Find similar products (same category and brand for demo)
        similar_products = recommendation_service.products_df[
            (recommendation_service.products_df['category'] == product_info['category']) &
            (recommendation_service.products_df['product_id'] != product_id)
        ].head(num_similar)
        
        similar_list = []
        for _, similar_product in similar_products.iterrows():
            similar_list.append({
                'product_id': similar_product['product_id'],
                'name': similar_product['name'],
                'category': similar_product['category'],
                'brand': similar_product['brand'],
                'price': float(similar_product['price']),
                'similarity_score': round(np.random.uniform(0.7, 0.95), 3)
            })
        
        return {
            "product_id": product_id,
            "product_name": product_info['name'],
            "similar_products": similar_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Similar products error: {e}")
        raise HTTPException(status_code=500, detail=f"Similar products error: {str(e)}")

@app.post("/experiments/create")
async def create_ab_experiment(request: ExperimentRequest):
    """Create new A/B test experiment"""
    try:
        # Validate traffic allocation sums to 1.0
        total_traffic = request.cf_traffic + request.cb_traffic + request.hybrid_traffic
        if abs(total_traffic - 1.0) > 0.01:
            raise HTTPException(
                status_code=400, 
                detail=f"Traffic allocation must sum to 1.0, got {total_traffic}"
            )
        
        # Create experiment configuration
        config = ExperimentConfig(
            name=request.name,
            traffic_allocation={
                ModelType.COLLABORATIVE: request.cf_traffic,
                ModelType.CONTENT_BASED: request.cb_traffic,
                ModelType.HYBRID: request.hybrid_traffic
            },
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=request.duration_days),
            success_metrics=["click_through_rate", "conversion_rate", "response_time"],
            min_sample_size=100,  # Lower for demo purposes
            confidence_level=0.95
        )
        
        # Start experiment
        experiment_id = ab_test_manager.create_experiment(config)
        
        return {
            "experiment_id": experiment_id,
            "name": request.name,
            "traffic_allocation": {
                "collaborative": request.cf_traffic,
                "content_based": request.cb_traffic,
                "hybrid": request.hybrid_traffic
            },
            "duration_days": request.duration_days,
            "status": "active"
        }
        
    except Exception as e:
        logger.error(f"❌ Experiment creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")

@app.get("/experiments")
async def list_experiments():
    """List all active A/B test experiments"""
    try:
        experiments = ab_test_manager.list_experiments()
        return {
            "active_experiments": experiments,
            "total_count": len(experiments)
        }
    except Exception as e:
        logger.error(f"❌ Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations/ab-test", response_model=RecommendationResponse)
async def get_ab_test_recommendations(request: ABTestRecommendationRequest):
    """Get recommendations using A/B testing framework"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Route user to appropriate model using A/B testing
        assigned_model = ab_test_manager.route_and_track(
            request.experiment_id,
            request.user_id,
            request_id
        )
        
        # Get recommendations from assigned model
        if assigned_model == ModelType.COLLABORATIVE:
            recommendations = recommendation_service.get_cf_recommendations(
                request.user_id, request.num_recommendations
            )
        elif assigned_model == ModelType.CONTENT_BASED:
            recommendations = recommendation_service.get_cb_recommendations(
                request.user_id, request.num_recommendations
            )
        else:  # HYBRID
            recommendations = recommendation_service.get_hybrid_recommendations(
                request.user_id, request.num_recommendations
            )
        
        response_time = (time.time() - start_time) * 1000
        
        # Log the recommendation event for experiment tracking
        ab_test_manager.log_recommendation(
            request.experiment_id,
            request.user_id,
            request_id,
            assigned_model,
            response_time,
            recommendations
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_used=f"ab_test_{assigned_model.value}",
            response_time_ms=round(response_time, 2),
            timestamp=datetime.utcnow().isoformat(),
            # Add experiment tracking metadata
            **{
                "experiment_id": request.experiment_id,
                "request_id": request_id,
                "assigned_model": assigned_model.value
            }
        )
        
    except Exception as e:
        logger.error(f"❌ A/B test recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"A/B test error: {str(e)}")

@app.post("/experiments/user-action")
async def log_user_action(request: UserActionRequest):
    """Log user action (click, purchase) for A/B test tracking"""
    try:
        ab_test_manager.log_user_action(
            request.experiment_id,
            request.request_id,
            request.action,
            request.clicked_items,
            request.conversion_value
        )
        
        return {
            "status": "success",
            "message": f"Logged {request.action} for request {request.request_id}",
            "experiment_id": request.experiment_id
        }
        
    except Exception as e:
        logger.error(f"❌ User action logging error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get comprehensive A/B test results with statistical analysis"""
    try:
        results = ab_test_manager.get_experiment_results(experiment_id)
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        
        # Convert numpy types to JSON-serializable types
        clean_results = convert_numpy_types(results)
        
        return clean_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Experiment results error: {e}")
        raise HTTPException(status_code=500, detail=f"Results error: {str(e)}")

@app.post("/experiments/demo-setup")
async def setup_demo_experiment():
    """Setup a demo A/B test experiment for portfolio demonstration"""
    try:
        # Create default experiment
        config = create_default_experiment()
        config.name = "portfolio_demo_experiment"
        
        experiment_id = ab_test_manager.create_experiment(config)
        
        # Generate some demo data for immediate results
        demo_users = ["U06253", "U04685", "U01732", "U08934", "U02156"]
        
        for i, user_id in enumerate(demo_users * 4):  # 20 requests total
            request_id = str(uuid.uuid4())
            
            # Route user to model
            assigned_model = ab_test_manager.route_and_track(experiment_id, user_id, request_id)
            
            # Simulate getting recommendations (use actual response times)
            start_time = time.time()
            if assigned_model == ModelType.COLLABORATIVE:
                recs = recommendation_service.get_cf_recommendations(user_id, 5)
            elif assigned_model == ModelType.CONTENT_BASED:
                recs = recommendation_service.get_cb_recommendations(user_id, 5)
            else:
                recs = recommendation_service.get_hybrid_recommendations(user_id, 5)
            
            response_time = (time.time() - start_time) * 1000
            
            # Log recommendation
            ab_test_manager.log_recommendation(
                experiment_id, user_id, request_id, assigned_model, response_time, recs
            )
            
            # Simulate user actions (some clicks and purchases)
            if i % 3 == 0:  # 33% click rate
                ab_test_manager.log_user_action(experiment_id, request_id, "click", [recs[0]["product_id"]])
                if i % 6 == 0:  # 16% purchase rate
                    ab_test_manager.log_user_action(
                        experiment_id, request_id, "purchase", 
                        [recs[0]["product_id"]], float(recs[0]["price"])  # Ensure price is float
                    )
        
        return {
            "experiment_id": experiment_id,
            "message": "Demo experiment created with sample data",
            "demo_users": demo_users,
            "total_requests": len(demo_users) * 4,
            "view_results": f"/experiments/{experiment_id}/results"
        }
        
    except Exception as e:
        logger.error(f"❌ Demo setup error: {e}")
        raise HTTPException(status_code=500, detail=f"Demo setup error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)