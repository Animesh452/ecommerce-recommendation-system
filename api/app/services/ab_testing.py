"""
A/B Testing Framework for Recommendation Models
Intelligent traffic routing and experiment tracking
"""

import random
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class ModelType(Enum):
    COLLABORATIVE = "cf"
    CONTENT_BASED = "cb"
    HYBRID = "hybrid"

@dataclass
class ExperimentConfig:
    """A/B test experiment configuration"""
    name: str
    traffic_allocation: Dict[ModelType, float]  # Model -> traffic percentage
    start_date: datetime
    end_date: datetime
    success_metrics: List[str]
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    
class TrafficRouter:
    """Intelligent traffic routing for A/B testing"""
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.allocation = experiment_config.traffic_allocation
        
        # Validate traffic allocation sums to 1.0
        total_allocation = sum(self.allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
    
    def route_request(self, user_id: str, request_id: str = None) -> ModelType:
        """Route user request to appropriate model based on A/B test config"""
        
        # Use deterministic routing based on user_id for consistency
        # Same user always gets same model during experiment
        user_hash = hash(user_id) % 10000
        
        # Calculate cumulative distribution
        cumulative = 0
        thresholds = {}
        
        for model, allocation in self.allocation.items():
            cumulative += allocation * 10000
            thresholds[model] = cumulative
        
        # Route based on hash
        for model, threshold in thresholds.items():
            if user_hash < threshold:
                logger.info(f"🎯 Routed user {user_id} to {model.value} (hash: {user_hash})")
                return model
        
        # Fallback (shouldn't happen with proper allocation)
        return ModelType.HYBRID

@dataclass
class ExperimentEvent:
    """Single event in A/B test experiment"""
    timestamp: datetime
    user_id: str
    request_id: str
    model_used: ModelType
    response_time_ms: float
    recommendations: List[Dict[str, Any]]
    user_action: Optional[str] = None  # 'click', 'purchase', 'ignore'
    clicked_items: List[str] = None
    conversion_value: float = 0.0

class ExperimentTracker:
    """Track and analyze A/B test experiments"""
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.events: List[ExperimentEvent] = []
        self.router = TrafficRouter(experiment_config)
    
    def log_recommendation_event(self, 
                                user_id: str,
                                request_id: str,
                                model_used: ModelType,
                                response_time_ms: float,
                                recommendations: List[Dict[str, Any]]) -> None:
        """Log a recommendation request event"""
        
        event = ExperimentEvent(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            request_id=request_id,
            model_used=model_used,
            response_time_ms=response_time_ms,
            recommendations=recommendations
        )
        
        self.events.append(event)
        logger.info(f"📊 Logged event: {model_used.value} for user {user_id}")
    
    def log_user_action(self,
                       request_id: str,
                       action: str,
                       clicked_items: List[str] = None,
                       conversion_value: float = 0.0) -> None:
        """Log user action (click, purchase) for conversion tracking"""
        
        # Find the corresponding recommendation event
        for event in self.events:
            if event.request_id == request_id:
                event.user_action = action
                event.clicked_items = clicked_items or []
                event.conversion_value = conversion_value
                logger.info(f"👆 User action: {action} for request {request_id}")
                break
    
    def get_experiment_results(self) -> Dict[str, Any]:
        """Calculate comprehensive experiment results"""
        
        if not self.events:
            return {"error": "No events recorded"}
        
        # Convert events to DataFrame for analysis
        data = []
        for event in self.events:
            data.append({
                'timestamp': event.timestamp,
                'user_id': event.user_id,
                'model': event.model_used.value,
                'response_time_ms': event.response_time_ms,
                'num_recommendations': len(event.recommendations),
                'user_action': event.user_action or 'no_action',
                'clicked_items_count': len(event.clicked_items) if event.clicked_items else 0,
                'conversion_value': event.conversion_value,
                'has_click': event.user_action == 'click',
                'has_purchase': event.user_action == 'purchase'
            })
        
        df = pd.DataFrame(data)
        
        # Calculate metrics by model
        results = {}
        
        for model in [ModelType.COLLABORATIVE, ModelType.CONTENT_BASED, ModelType.HYBRID]:
            model_data = df[df['model'] == model.value]
            
            if len(model_data) == 0:
                continue
            
            # Performance metrics
            metrics = {
                'total_requests': len(model_data),
                'avg_response_time_ms': model_data['response_time_ms'].mean(),
                'p95_response_time_ms': model_data['response_time_ms'].quantile(0.95),
                'click_through_rate': model_data['has_click'].mean(),
                'conversion_rate': model_data['has_purchase'].mean(),
                'avg_conversion_value': model_data['conversion_value'].mean(),
                'total_revenue': model_data['conversion_value'].sum(),
                'traffic_percentage': len(model_data) / len(df) * 100
            }
            
            results[model.value] = metrics
        
        # Statistical significance testing
        significance_tests = self._run_significance_tests(df)
        
        # Overall experiment summary
        summary = {
            'experiment_name': self.config.name,
            'start_date': self.config.start_date.isoformat(),
            'total_events': len(df),
            'experiment_duration_hours': (datetime.utcnow() - self.config.start_date).total_seconds() / 3600,
            'models_tested': list(results.keys()),
            'winning_model': self._determine_winner(results),
            'statistical_significance': significance_tests
        }
        
        return {
            'summary': summary,
            'model_results': results,
            'raw_data_sample': df.head(10).to_dict('records')
        }
    
    def _run_significance_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run statistical significance tests between models"""
        
        models = df['model'].unique()
        if len(models) < 2:
            return {"error": "Need at least 2 models for significance testing"}
        
        tests = {}
        
        # CTR comparison (Chi-square test)
        try:
            contingency_table = pd.crosstab(df['model'], df['has_click'])
            chi2, p_value_ctr, dof, expected = stats.chi2_contingency(contingency_table)
            
            tests['click_through_rate'] = {
                'test': 'chi_square',
                'chi2_statistic': chi2,
                'p_value': p_value_ctr,
                'significant': p_value_ctr < (1 - self.config.confidence_level)
            }
        except Exception as e:
            tests['click_through_rate'] = {"error": str(e)}
        
        # Response time comparison (ANOVA)
        try:
            model_groups = [df[df['model'] == model]['response_time_ms'].values for model in models]
            f_stat, p_value_time = stats.f_oneway(*model_groups)
            
            tests['response_time'] = {
                'test': 'anova',
                'f_statistic': f_stat,
                'p_value': p_value_time,
                'significant': p_value_time < (1 - self.config.confidence_level)
            }
        except Exception as e:
            tests['response_time'] = {"error": str(e)}
        
        return tests
    
    def _determine_winner(self, results: Dict[str, Dict]) -> str:
        """Determine winning model based on business metrics"""
        
        if not results:
            return "insufficient_data"
        
        # Weighted scoring: CTR (40%) + Conversion Rate (40%) + Response Time (20%)
        scores = {}
        
        for model, metrics in results.items():
            if metrics['total_requests'] < self.config.min_sample_size:
                continue
            
            # Normalize metrics (higher is better for CTR and conversion, lower is better for response time)
            ctr_score = metrics['click_through_rate'] * 100
            conv_score = metrics['conversion_rate'] * 100
            time_score = max(0, 100 - metrics['avg_response_time_ms'])  # Penalty for slow response
            
            weighted_score = (ctr_score * 0.4) + (conv_score * 0.4) + (time_score * 0.2)
            scores[model] = weighted_score
        
        if not scores:
            return "insufficient_sample_size"
        
        winner = max(scores.items(), key=lambda x: x[1])
        return winner[0]

class ABTestManager:
    """High-level A/B test management"""
    
    def __init__(self):
        self.active_experiments: Dict[str, ExperimentTracker] = {}
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create and start new A/B test experiment"""
        
        experiment_id = f"{config.name}_{int(time.time())}"
        tracker = ExperimentTracker(config)
        self.active_experiments[experiment_id] = tracker
        
        logger.info(f"🧪 Started A/B test: {experiment_id}")
        return experiment_id
    
    def route_and_track(self, 
                       experiment_id: str,
                       user_id: str,
                       request_id: str = None) -> ModelType:
        """Route user request and prepare for tracking"""
        
        if experiment_id not in self.active_experiments:
            logger.warning(f"Experiment {experiment_id} not found, using hybrid")
            return ModelType.HYBRID
        
        tracker = self.active_experiments[experiment_id]
        return tracker.router.route_request(user_id, request_id)
    
    def log_recommendation(self,
                          experiment_id: str,
                          user_id: str,
                          request_id: str,
                          model_used: ModelType,
                          response_time_ms: float,
                          recommendations: List[Dict[str, Any]]) -> None:
        """Log recommendation event for experiment tracking"""
        
        if experiment_id in self.active_experiments:
            self.active_experiments[experiment_id].log_recommendation_event(
                user_id, request_id, model_used, response_time_ms, recommendations
            )
    
    def log_user_action(self,
                       experiment_id: str,
                       request_id: str,
                       action: str,
                       clicked_items: List[str] = None,
                       conversion_value: float = 0.0) -> None:
        """Log user action for conversion tracking"""
        
        if experiment_id in self.active_experiments:
            self.active_experiments[experiment_id].log_user_action(
                request_id, action, clicked_items, conversion_value
            )
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive results for an experiment"""
        
        if experiment_id not in self.active_experiments:
            return {"error": f"Experiment {experiment_id} not found"}
        
        return self.active_experiments[experiment_id].get_experiment_results()
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all active experiments"""
        
        experiments = []
        for exp_id, tracker in self.active_experiments.items():
            experiments.append({
                'experiment_id': exp_id,
                'name': tracker.config.name,
                'start_date': tracker.config.start_date.isoformat(),
                'traffic_allocation': {model.value: allocation for model, allocation in tracker.config.traffic_allocation.items()},
                'total_events': len(tracker.events)
            })
        
        return experiments

# Example usage and default experiment setup
def create_default_experiment() -> ExperimentConfig:
    """Create default A/B test comparing all three models"""
    
    return ExperimentConfig(
        name="model_comparison_v1",
        traffic_allocation={
            ModelType.COLLABORATIVE: 0.3,    # 30% traffic
            ModelType.CONTENT_BASED: 0.3,    # 30% traffic  
            ModelType.HYBRID: 0.4            # 40% traffic (hypothesis: best performer)
        },
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow() + timedelta(days=7),
        success_metrics=["click_through_rate", "conversion_rate", "response_time"],
        min_sample_size=500,
        confidence_level=0.95
    )

# Global A/B test manager instance
ab_test_manager = ABTestManager()