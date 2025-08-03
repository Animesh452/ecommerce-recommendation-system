"""
MLOps Monitoring and Automation System
Real-time performance tracking, automated retraining, and cost optimization
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from collections import deque
import threading
import logging
import os

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class MetricType(Enum):
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    MODEL_ACCURACY = "model_accuracy"
    BUSINESS_REVENUE = "business_revenue"
    CONVERSION_RATE = "conversion_rate"

@dataclass
class MetricThreshold:
    """Performance threshold configuration"""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    window_minutes: int = 5
    min_samples: int = 10

@dataclass
class PerformanceAlert:
    """Performance alert object"""
    timestamp: datetime
    metric_type: MetricType
    severity: AlertSeverity
    current_value: float
    threshold_value: float
    message: str
    model_name: Optional[str] = None

class ModelPerformanceTracker:
    """Track real-time model performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = {
            'response_times': deque(maxlen=window_size),
            'error_counts': deque(maxlen=window_size),
            'request_counts': deque(maxlen=window_size),
            'model_predictions': deque(maxlen=window_size),
            'business_metrics': deque(maxlen=window_size),
            'timestamps': deque(maxlen=window_size)
        }
        
        # Model-specific tracking
        self.model_metrics = {
            'cf': {'response_times': deque(maxlen=window_size), 'accuracy_scores': deque(maxlen=window_size)},
            'cb': {'response_times': deque(maxlen=window_size), 'accuracy_scores': deque(maxlen=window_size)},
            'hybrid': {'response_times': deque(maxlen=window_size), 'accuracy_scores': deque(maxlen=window_size)}
        }
        
        # Performance thresholds
        self.thresholds = [
            MetricThreshold(MetricType.RESPONSE_TIME, 100.0, 200.0, 5, 10),  # ms
            MetricThreshold(MetricType.ERROR_RATE, 0.05, 0.10, 5, 10),       # 5% warning, 10% critical
            MetricThreshold(MetricType.THROUGHPUT, 50.0, 20.0, 5, 10),       # requests/min
            MetricThreshold(MetricType.CONVERSION_RATE, 0.10, 0.05, 60, 50), # hourly conversion rate
        ]
        
        self.alerts = []
        
    def log_request(self, 
                   model_name: str,
                   response_time_ms: float,
                   success: bool = True,
                   user_action: str = None,
                   conversion_value: float = 0.0):
        """Log a single request for performance tracking"""
        
        timestamp = datetime.utcnow()
        
        # Global metrics
        self.metrics['response_times'].append(response_time_ms)
        self.metrics['error_counts'].append(0 if success else 1)
        self.metrics['request_counts'].append(1)
        self.metrics['timestamps'].append(timestamp)
        
        # Business metrics
        business_metric = {
            'conversion_value': conversion_value,
            'has_conversion': conversion_value > 0,
            'user_action': user_action or 'no_action'
        }
        self.metrics['business_metrics'].append(business_metric)
        
        # Model-specific metrics
        if model_name in self.model_metrics:
            self.model_metrics[model_name]['response_times'].append(response_time_ms)
            # Simulate accuracy score (in production, this would be calculated from user feedback)
            accuracy_score = np.random.uniform(0.7, 0.95) if success else 0.0
            self.model_metrics[model_name]['accuracy_scores'].append(accuracy_score)
        
        # Check thresholds
        self._check_thresholds()
        
        logger.info(f"📊 Logged request: {model_name}, {response_time_ms:.2f}ms, success={success}")
    
    def _check_thresholds(self):
        """Check if any performance thresholds are breached"""
        
        if len(self.metrics['timestamps']) < 10:
            return  # Need minimum samples
        
        current_time = datetime.utcnow()
        
        for threshold in self.thresholds:
            try:
                # Get recent data within time window
                cutoff_time = current_time - timedelta(minutes=threshold.window_minutes)
                recent_indices = [
                    i for i, ts in enumerate(self.metrics['timestamps'])
                    if ts >= cutoff_time
                ]
                
                if len(recent_indices) < threshold.min_samples:
                    continue
                
                # Calculate metric value
                if threshold.metric_type == MetricType.RESPONSE_TIME:
                    recent_times = [self.metrics['response_times'][i] for i in recent_indices]
                    current_value = np.mean(recent_times)
                    
                elif threshold.metric_type == MetricType.ERROR_RATE:
                    recent_errors = [self.metrics['error_counts'][i] for i in recent_indices]
                    current_value = np.mean(recent_errors)
                    
                elif threshold.metric_type == MetricType.THROUGHPUT:
                    requests_in_window = len(recent_indices)
                    current_value = requests_in_window / threshold.window_minutes
                    
                elif threshold.metric_type == MetricType.CONVERSION_RATE:
                    recent_business = [self.metrics['business_metrics'][i] for i in recent_indices]
                    conversions = sum(1 for metric in recent_business if metric['has_conversion'])
                    current_value = conversions / len(recent_business) if recent_business else 0
                
                else:
                    continue
                
                # Check thresholds
                severity = None
                threshold_value = None
                
                if current_value >= threshold.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif current_value >= threshold.warning_threshold:
                    severity = AlertSeverity.WARNING
                    threshold_value = threshold.warning_threshold
                
                # Create alert if threshold breached
                if severity:
                    alert = PerformanceAlert(
                        timestamp=current_time,
                        metric_type=threshold.metric_type,
                        severity=severity,
                        current_value=current_value,
                        threshold_value=threshold_value,
                        message=f"{threshold.metric_type.value} {severity.value}: {current_value:.3f} exceeds {threshold_value}"
                    )
                    
                    self.alerts.append(alert)
                    logger.warning(f"🚨 {alert.message}")
                    
            except Exception as e:
                logger.error(f"❌ Threshold check error for {threshold.metric_type}: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics summary"""
        
        if not self.metrics['timestamps']:
            return {"error": "No metrics available"}
        
        # Calculate recent performance (last 5 minutes)
        current_time = datetime.utcnow()
        recent_cutoff = current_time - timedelta(minutes=5)
        
        recent_indices = [
            i for i, ts in enumerate(self.metrics['timestamps'])
            if ts >= recent_cutoff
        ]
        
        if not recent_indices:
            recent_indices = list(range(max(0, len(self.metrics['timestamps']) - 10), 
                                      len(self.metrics['timestamps'])))
        
        # Global metrics
        recent_response_times = [self.metrics['response_times'][i] for i in recent_indices]
        recent_errors = [self.metrics['error_counts'][i] for i in recent_indices]
        recent_business = [self.metrics['business_metrics'][i] for i in recent_indices]
        
        global_metrics = {
            'avg_response_time_ms': np.mean(recent_response_times) if recent_response_times else 0,
            'p95_response_time_ms': np.percentile(recent_response_times, 95) if recent_response_times else 0,
            'error_rate': np.mean(recent_errors) if recent_errors else 0,
            'requests_per_minute': len(recent_indices) / 5.0,  # 5-minute window
            'conversion_rate': np.mean([m['has_conversion'] for m in recent_business]) if recent_business else 0,
            'total_revenue': sum(m['conversion_value'] for m in recent_business),
        }
        
        # Model-specific metrics
        model_metrics = {}
        for model_name, model_data in self.model_metrics.items():
            if model_data['response_times']:
                model_metrics[model_name] = {
                    'avg_response_time_ms': np.mean(list(model_data['response_times'])[-20:]),
                    'avg_accuracy_score': np.mean(list(model_data['accuracy_scores'])[-20:]),
                    'total_requests': len(model_data['response_times'])
                }
        
        # Recent alerts
        recent_alerts = [
            asdict(alert) for alert in self.alerts[-10:]
        ]
        
        return {
            'timestamp': current_time.isoformat(),
            'global_metrics': global_metrics,
            'model_metrics': model_metrics,
            'recent_alerts': recent_alerts,
            'total_requests': len(self.metrics['timestamps']),
            'monitoring_window_minutes': 5
        }

class AutoRetrainingManager:
    """Manage automated model retraining based on performance degradation"""
    
    def __init__(self, performance_tracker: ModelPerformanceTracker):
        self.tracker = performance_tracker
        self.retraining_thresholds = {
            'accuracy_drop': 0.1,          # 10% accuracy drop triggers retraining
            'response_time_increase': 2.0,  # 2x response time increase
            'error_rate_spike': 0.15,       # 15% error rate
            'min_days_between_retraining': 1
        }
        self.last_retraining = {
            'cf': datetime.utcnow() - timedelta(days=30),
            'cb': datetime.utcnow() - timedelta(days=30),
            'hybrid': datetime.utcnow() - timedelta(days=30)
        }
        self.retraining_jobs = []
    
    def check_retraining_triggers(self) -> List[str]:
        """Check if any models need retraining"""
        
        models_needing_retraining = []
        current_metrics = self.tracker.get_current_metrics()
        
        if 'model_metrics' not in current_metrics:
            return models_needing_retraining
        
        for model_name, metrics in current_metrics['model_metrics'].items():
            needs_retraining = False
            reason = []
            
            # Check if enough time has passed since last retraining
            days_since_retrain = (datetime.utcnow() - self.last_retraining[model_name]).days
            if days_since_retrain < self.retraining_thresholds['min_days_between_retraining']:
                continue
            
            # Check accuracy degradation
            if metrics.get('avg_accuracy_score', 1.0) < 0.8:  # Below 80% accuracy
                needs_retraining = True
                reason.append(f"accuracy dropped to {metrics['avg_accuracy_score']:.2%}")
            
            # Check response time increase
            if metrics.get('avg_response_time_ms', 0) > 100:  # Above 100ms average
                needs_retraining = True
                reason.append(f"response time increased to {metrics['avg_response_time_ms']:.1f}ms")
            
            if needs_retraining:
                models_needing_retraining.append({
                    'model': model_name,
                    'reasons': reason,
                    'current_accuracy': metrics.get('avg_accuracy_score', 0),
                    'current_response_time': metrics.get('avg_response_time_ms', 0)
                })
                
                logger.info(f"🔄 Model {model_name} flagged for retraining: {', '.join(reason)}")
        
        return models_needing_retraining
    
    def trigger_retraining(self, model_name: str) -> Dict[str, Any]:
        """Trigger automated retraining for a specific model"""
        
        job_id = f"retrain_{model_name}_{int(time.time())}"
        
        # In production, this would trigger actual retraining pipeline
        # For demo, we simulate the retraining process
        retraining_job = {
            'job_id': job_id,
            'model_name': model_name,
            'status': 'initiated',
            'start_time': datetime.utcnow().isoformat(),
            'estimated_duration_minutes': 30,
            'progress': 0,
            'stage': 'data_preparation'
        }
        
        self.retraining_jobs.append(retraining_job)
        self.last_retraining[model_name] = datetime.utcnow()
        
        logger.info(f"🔄 Started retraining job {job_id} for model {model_name}")
        
        return retraining_job
    
    def get_retraining_status(self) -> List[Dict[str, Any]]:
        """Get status of all retraining jobs"""
        
        # Simulate job progress for demo
        for job in self.retraining_jobs:
            if job['status'] == 'initiated':
                start_time = datetime.fromisoformat(job['start_time'])
                elapsed_minutes = (datetime.utcnow() - start_time).total_seconds() / 60
                
                if elapsed_minutes < 5:
                    job['progress'] = min(20, elapsed_minutes * 4)
                    job['stage'] = 'data_preparation'
                elif elapsed_minutes < 15:
                    job['progress'] = min(60, 20 + (elapsed_minutes - 5) * 4)
                    job['stage'] = 'model_training'
                elif elapsed_minutes < 25:
                    job['progress'] = min(90, 60 + (elapsed_minutes - 15) * 3)
                    job['stage'] = 'model_validation'
                else:
                    job['progress'] = 100
                    job['stage'] = 'deployment'
                    job['status'] = 'completed'
        
        return self.retraining_jobs

class CostOptimizationManager:
    """Manage cost optimization and resource scaling"""
    
    def __init__(self, performance_tracker: ModelPerformanceTracker):
        self.tracker = performance_tracker
        self.cost_metrics = {
            'container_cost_per_hour': 0.10,     # $0.10/hour per container
            'request_cost': 0.001,               # $0.001 per request
            'storage_cost_per_gb_month': 0.023,  # S3 storage cost
            'data_transfer_cost_per_gb': 0.09    # Data transfer cost
        }
        
    def calculate_current_costs(self) -> Dict[str, Any]:
        """Calculate current operational costs"""
        
        metrics = self.tracker.get_current_metrics()
        
        if 'global_metrics' not in metrics:
            return {"error": "No metrics available for cost calculation"}
        
        # Calculate hourly costs
        requests_per_hour = metrics['global_metrics']['requests_per_minute'] * 60
        
        costs = {
            'hourly_container_cost': 2 * self.cost_metrics['container_cost_per_hour'],  # 2 containers
            'hourly_request_cost': requests_per_hour * self.cost_metrics['request_cost'],
            'estimated_daily_cost': (2 * self.cost_metrics['container_cost_per_hour'] * 24 + 
                                   requests_per_hour * self.cost_metrics['request_cost'] * 24),
            'estimated_monthly_cost': None  # Will calculate below
        }
        
        costs['estimated_monthly_cost'] = costs['estimated_daily_cost'] * 30
        
        # Cost efficiency metrics
        revenue_per_hour = metrics['global_metrics']['total_revenue'] * 12  # Scale to hourly
        cost_efficiency = {
            'revenue_per_hour': revenue_per_hour,
            'cost_per_hour': costs['hourly_container_cost'] + costs['hourly_request_cost'],
            'roi_ratio': revenue_per_hour / (costs['hourly_container_cost'] + costs['hourly_request_cost']) if costs['hourly_container_cost'] + costs['hourly_request_cost'] > 0 else 0,
            'cost_per_request': (costs['hourly_container_cost'] + costs['hourly_request_cost']) / max(requests_per_hour, 1)
        }
        
        # Optimization recommendations
        recommendations = []
        
        if cost_efficiency['cost_per_request'] > 0.01:
            recommendations.append("High cost per request - consider optimizing response times")
        
        if requests_per_hour < 20:
            recommendations.append("Low traffic - consider scaling down containers during off-peak hours")
        
        if cost_efficiency['roi_ratio'] < 2.0:
            recommendations.append("Low ROI - focus on improving conversion rates")
        
        return {
            'current_costs': costs,
            'efficiency_metrics': cost_efficiency,
            'optimization_recommendations': recommendations,
            'timestamp': datetime.utcnow().isoformat()
        }

# Global instances
performance_tracker = ModelPerformanceTracker()
retraining_manager = AutoRetrainingManager(performance_tracker)
cost_manager = CostOptimizationManager(performance_tracker)