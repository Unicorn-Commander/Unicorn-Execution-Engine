#!/usr/bin/env python3
"""
Gemma 3n E4B Elastic Parameter Activation System
Dynamic scaling system for elastic parameter activation/deactivation
"""

import os
import sys
import time
import json
import logging
import numpy as np
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElasticState(Enum):
    """Elastic parameter states"""
    INACTIVE = "inactive"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DEACTIVATING = "deactivating"
    FAILED = "failed"

@dataclass
class ElasticParameter:
    """Represents an elastic parameter with its metadata"""
    layer_id: int
    parameter_id: str
    parameter_type: str
    memory_requirement: int
    compute_requirement: float
    priority: int
    activation_cost: float
    deactivation_cost: float
    quality_impact: float
    state: ElasticState
    hardware_target: str
    last_used: float
    activation_count: int

@dataclass
class WorkloadMetrics:
    """Real-time workload metrics"""
    current_load: float
    avg_load: float
    peak_load: float
    memory_pressure: float
    compute_pressure: float
    queue_length: int
    response_time: float
    throughput: float
    error_rate: float
    timestamp: float

@dataclass
class ActivationDecision:
    """Decision for elastic parameter activation/deactivation"""
    parameter_id: str
    action: str  # "activate" or "deactivate"
    urgency: float
    expected_impact: float
    cost: float
    reason: str

class Gemma3nE4BElasticActivationSystem:
    """Dynamic elastic parameter activation system for Gemma 3n E4B"""
    
    def __init__(self, model_path: str = "./models/gemma-3n-e4b-it"):
        self.model_path = Path(model_path)
        
        # System configuration
        self.config = {
            "base_parameters": 2 * 1024**3,  # 2B base parameters
            "elastic_parameters": 2 * 1024**3,  # 2B elastic parameters
            "max_active_elastic": 1.5 * 1024**3,  # 1.5B max active elastic
            "min_active_elastic": 0.5 * 1024**3,  # 0.5B min active elastic
            "activation_threshold": 0.7,  # Activate when load > 70%
            "deactivation_threshold": 0.3,  # Deactivate when load < 30%
            "response_time_target": 100,  # 100ms target response time
            "throughput_target": 100,  # 100 TPS target
            "quality_threshold": 0.95,  # 95% quality preservation
            "memory_safety_margin": 0.1,  # 10% memory safety margin
            "decision_interval": 1.0,  # 1 second decision interval
            "warmup_time": 5.0,  # 5 seconds warmup time
            "cooldown_time": 10.0,  # 10 seconds cooldown time
            "max_activation_rate": 0.2,  # 20% max activation rate per interval
            "max_deactivation_rate": 0.1  # 10% max deactivation rate per interval
        }
        
        # Hardware configurations
        self.hardware_config = {
            "npu_phoenix": {
                "memory_budget": 2 * 1024**3,  # 2GB NPU SRAM
                "compute_budget": 16 * 1024**3,  # 16 TOPS
                "activation_latency": 0.001,  # 1ms activation latency
                "preferred_types": ["attention", "embedding"]
            },
            "radeon_780m": {
                "memory_budget": 16 * 1024**3,  # 16GB DDR5
                "compute_budget": 2.7 * 1024**3,  # 2.7 TFLOPS
                "activation_latency": 0.005,  # 5ms activation latency
                "preferred_types": ["ffn", "projection"]
            },
            "system_memory": {
                "memory_budget": 80 * 1024**3,  # 80GB DDR5
                "compute_budget": 1.2 * 1024**3,  # 1.2 TFLOPS
                "activation_latency": 0.010,  # 10ms activation latency
                "preferred_types": ["backup", "storage"]
            }
        }
        
        # Elastic parameters registry
        self.elastic_parameters = {}
        self.active_parameters = set()
        self.inactive_parameters = set()
        self.activating_parameters = set()
        self.deactivating_parameters = set()
        
        # Monitoring and metrics
        self.workload_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self.decision_history = deque(maxlen=500)   # Keep last 500 decisions
        self.performance_history = deque(maxlen=200)  # Keep last 200 performance metrics
        
        # Control flags
        self.system_active = False
        self.monitoring_thread = None
        self.decision_thread = None
        
        # Callbacks
        self.activation_callbacks = []
        self.deactivation_callbacks = []
        self.metrics_callbacks = []
        
        # Initialize system
        self.initialize_elastic_parameters()
        
    def initialize_elastic_parameters(self):
        """Initialize elastic parameters based on Gemma 3n E4B architecture"""
        logger.info("🔧 Initializing elastic parameters...")
        
        # Gemma 3n E4B layer configuration
        num_layers = 24
        hidden_size = 3072
        num_heads = 24
        intermediate_size = 8192
        
        parameter_id = 0
        
        for layer_idx in range(num_layers):
            # Elastic attention parameters
            attention_params = [
                ("q_proj_elastic", "attention", hidden_size * hidden_size // 2),
                ("k_proj_elastic", "attention", hidden_size * hidden_size // 2),
                ("v_proj_elastic", "attention", hidden_size * hidden_size // 2),
                ("o_proj_elastic", "attention", hidden_size * hidden_size // 2)
            ]
            
            for param_name, param_type, param_size in attention_params:
                param_id = f"layer_{layer_idx}_{param_name}"
                
                # Determine optimal hardware target
                hardware_target = self.select_optimal_hardware(param_type, param_size)
                
                elastic_param = ElasticParameter(
                    layer_id=layer_idx,
                    parameter_id=param_id,
                    parameter_type=param_type,
                    memory_requirement=param_size * 1,  # INT8 = 1 byte
                    compute_requirement=param_size * 2,  # Estimate compute
                    priority=self.calculate_parameter_priority(layer_idx, param_type),
                    activation_cost=self.calculate_activation_cost(param_size, hardware_target),
                    deactivation_cost=self.calculate_deactivation_cost(param_size, hardware_target),
                    quality_impact=self.estimate_quality_impact(layer_idx, param_type),
                    state=ElasticState.INACTIVE,
                    hardware_target=hardware_target,
                    last_used=0.0,
                    activation_count=0
                )
                
                self.elastic_parameters[param_id] = elastic_param
                self.inactive_parameters.add(param_id)
                parameter_id += 1
            
            # Elastic FFN parameters
            ffn_params = [
                ("gate_proj_elastic", "ffn", hidden_size * intermediate_size // 2),
                ("up_proj_elastic", "ffn", hidden_size * intermediate_size // 2),
                ("down_proj_elastic", "ffn", intermediate_size * hidden_size // 2)
            ]
            
            for param_name, param_type, param_size in ffn_params:
                param_id = f"layer_{layer_idx}_{param_name}"
                
                # Determine optimal hardware target
                hardware_target = self.select_optimal_hardware(param_type, param_size)
                
                elastic_param = ElasticParameter(
                    layer_id=layer_idx,
                    parameter_id=param_id,
                    parameter_type=param_type,
                    memory_requirement=param_size * 0.5,  # INT4 = 0.5 bytes
                    compute_requirement=param_size * 1.5,  # Estimate compute
                    priority=self.calculate_parameter_priority(layer_idx, param_type),
                    activation_cost=self.calculate_activation_cost(param_size, hardware_target),
                    deactivation_cost=self.calculate_deactivation_cost(param_size, hardware_target),
                    quality_impact=self.estimate_quality_impact(layer_idx, param_type),
                    state=ElasticState.INACTIVE,
                    hardware_target=hardware_target,
                    last_used=0.0,
                    activation_count=0
                )
                
                self.elastic_parameters[param_id] = elastic_param
                self.inactive_parameters.add(param_id)
                parameter_id += 1
        
        logger.info(f"   ✅ Initialized {len(self.elastic_parameters)} elastic parameters")
        logger.info(f"   📊 Total elastic memory: {sum(p.memory_requirement for p in self.elastic_parameters.values()) / 1024**3:.1f}GB")
        
    def select_optimal_hardware(self, param_type: str, param_size: int) -> str:
        """Select optimal hardware target for parameter"""
        
        # Attention parameters prefer NPU
        if param_type == "attention":
            return "npu_phoenix"
        
        # FFN parameters prefer iGPU
        elif param_type == "ffn":
            return "radeon_780m"
        
        # Default to system memory
        else:
            return "system_memory"
    
    def calculate_parameter_priority(self, layer_idx: int, param_type: str) -> int:
        """Calculate priority for parameter activation"""
        
        # Early and late layers get higher priority
        if layer_idx < 4 or layer_idx > 20:
            base_priority = 1
        else:
            base_priority = 2
        
        # Attention parameters get higher priority
        if param_type == "attention":
            return base_priority
        else:
            return base_priority + 1
    
    def calculate_activation_cost(self, param_size: int, hardware_target: str) -> float:
        """Calculate cost of activating parameter"""
        
        # Base cost proportional to size
        base_cost = param_size / 1024**3  # Cost per GB
        
        # Hardware-specific multipliers
        hardware_multipliers = {
            "npu_phoenix": 1.0,    # NPU is efficient
            "radeon_780m": 1.2,    # iGPU has some overhead
            "system_memory": 1.5   # System memory has highest overhead
        }
        
        return base_cost * hardware_multipliers.get(hardware_target, 1.0)
    
    def calculate_deactivation_cost(self, param_size: int, hardware_target: str) -> float:
        """Calculate cost of deactivating parameter"""
        
        # Deactivation is generally cheaper than activation
        activation_cost = self.calculate_activation_cost(param_size, hardware_target)
        return activation_cost * 0.3  # 30% of activation cost
    
    def estimate_quality_impact(self, layer_idx: int, param_type: str) -> float:
        """Estimate quality impact of parameter activation"""
        
        # Middle layers have higher quality impact
        if 8 <= layer_idx <= 16:
            base_impact = 0.8
        else:
            base_impact = 0.6
        
        # Attention parameters have higher quality impact
        if param_type == "attention":
            return base_impact
        else:
            return base_impact * 0.7
    
    def start_monitoring(self):
        """Start the monitoring and decision-making system"""
        logger.info("🚀 Starting elastic parameter activation system...")
        
        self.system_active = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start decision thread
        self.decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        self.decision_thread.start()
        
        logger.info("   ✅ Monitoring system started")
        logger.info("   ✅ Decision system started")
    
    def stop_monitoring(self):
        """Stop the monitoring and decision-making system"""
        logger.info("🛑 Stopping elastic parameter activation system...")
        
        self.system_active = False
        
        # Wait for threads to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        if self.decision_thread:
            self.decision_thread.join(timeout=5.0)
        
        logger.info("   ✅ System stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.system_active:
            try:
                # Collect workload metrics
                metrics = self.collect_workload_metrics()
                
                # Store metrics
                self.workload_history.append(metrics)
                
                # Trigger callbacks
                for callback in self.metrics_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Metrics callback error: {e}")
                
                # Sleep for next measurement
                time.sleep(0.1)  # 100ms monitoring interval
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _decision_loop(self):
        """Main decision-making loop"""
        while self.system_active:
            try:
                # Make activation/deactivation decisions
                decisions = self.make_activation_decisions()
                
                # Execute decisions
                for decision in decisions:
                    self.execute_decision(decision)
                
                # Sleep for next decision cycle
                time.sleep(self.config["decision_interval"])
                
            except Exception as e:
                logger.error(f"Decision loop error: {e}")
                time.sleep(1.0)
    
    def collect_workload_metrics(self) -> WorkloadMetrics:
        """Collect current workload metrics"""
        
        # Simulate workload metrics (in real implementation, this would collect from actual system)
        current_time = time.time()
        
        # Generate simulated metrics
        base_load = 0.5 + 0.3 * np.sin(current_time * 0.1)  # Oscillating load
        noise = np.random.normal(0, 0.05)  # Small random noise
        current_load = max(0.0, min(1.0, base_load + noise))
        
        # Calculate derived metrics
        recent_metrics = list(self.workload_history)[-10:]  # Last 10 metrics
        if recent_metrics:
            avg_load = np.mean([m.current_load for m in recent_metrics])
            peak_load = np.max([m.current_load for m in recent_metrics])
        else:
            avg_load = current_load
            peak_load = current_load
        
        # Memory and compute pressure
        active_memory = sum(
            self.elastic_parameters[pid].memory_requirement 
            for pid in self.active_parameters
        )
        total_memory_budget = sum(hw["memory_budget"] for hw in self.hardware_config.values())
        memory_pressure = active_memory / total_memory_budget
        
        active_compute = sum(
            self.elastic_parameters[pid].compute_requirement 
            for pid in self.active_parameters
        )
        total_compute_budget = sum(hw["compute_budget"] for hw in self.hardware_config.values())
        compute_pressure = active_compute / total_compute_budget
        
        # Response time and throughput (simulated)
        response_time = 50 + 100 * current_load + np.random.normal(0, 10)
        throughput = 150 * (1 - current_load) + np.random.normal(0, 5)
        
        metrics = WorkloadMetrics(
            current_load=current_load,
            avg_load=avg_load,
            peak_load=peak_load,
            memory_pressure=memory_pressure,
            compute_pressure=compute_pressure,
            queue_length=int(current_load * 50),  # Simulated queue length
            response_time=max(10, response_time),
            throughput=max(10, throughput),
            error_rate=max(0, current_load - 0.8) * 0.1,  # Errors when overloaded
            timestamp=current_time
        )
        
        return metrics
    
    def make_activation_decisions(self) -> List[ActivationDecision]:
        """Make decisions about parameter activation/deactivation"""
        
        if not self.workload_history:
            return []
        
        # Get current metrics
        current_metrics = self.workload_history[-1]
        
        # Determine if we need to activate or deactivate
        decisions = []
        
        # Check for activation conditions
        if self.should_activate_parameters(current_metrics):
            activation_decisions = self.select_parameters_to_activate(current_metrics)
            decisions.extend(activation_decisions)
        
        # Check for deactivation conditions
        if self.should_deactivate_parameters(current_metrics):
            deactivation_decisions = self.select_parameters_to_deactivate(current_metrics)
            decisions.extend(deactivation_decisions)
        
        # Store decisions
        for decision in decisions:
            self.decision_history.append(decision)
        
        return decisions
    
    def should_activate_parameters(self, metrics: WorkloadMetrics) -> bool:
        """Determine if parameters should be activated"""
        
        # Check activation conditions
        conditions = [
            metrics.current_load > self.config["activation_threshold"],
            metrics.response_time > self.config["response_time_target"],
            metrics.throughput < self.config["throughput_target"],
            metrics.error_rate > 0.01  # 1% error rate
        ]
        
        # Need at least 2 conditions to activate
        return sum(conditions) >= 2
    
    def should_deactivate_parameters(self, metrics: WorkloadMetrics) -> bool:
        """Determine if parameters should be deactivated"""
        
        # Check deactivation conditions
        conditions = [
            metrics.current_load < self.config["deactivation_threshold"],
            metrics.response_time < self.config["response_time_target"] * 0.7,
            metrics.throughput > self.config["throughput_target"] * 1.3,
            metrics.memory_pressure > 0.9  # Memory pressure too high
        ]
        
        # Need at least 2 conditions to deactivate
        return sum(conditions) >= 2
    
    def select_parameters_to_activate(self, metrics: WorkloadMetrics) -> List[ActivationDecision]:
        """Select parameters to activate based on current metrics"""
        
        decisions = []
        
        # Get candidate parameters for activation
        candidates = [
            (pid, self.elastic_parameters[pid]) 
            for pid in self.inactive_parameters
        ]
        
        # Sort by priority and quality impact
        candidates.sort(key=lambda x: (x[1].priority, -x[1].quality_impact))
        
        # Calculate how many parameters to activate
        current_active = len(self.active_parameters)
        max_additional = int(len(self.inactive_parameters) * self.config["max_activation_rate"])
        
        activated_count = 0
        
        for param_id, param in candidates:
            if activated_count >= max_additional:
                break
            
            # Check if we can afford to activate this parameter
            if self.can_afford_activation(param, metrics):
                decision = ActivationDecision(
                    parameter_id=param_id,
                    action="activate",
                    urgency=self.calculate_activation_urgency(param, metrics),
                    expected_impact=param.quality_impact,
                    cost=param.activation_cost,
                    reason=f"Load: {metrics.current_load:.2f}, RT: {metrics.response_time:.1f}ms"
                )
                decisions.append(decision)
                activated_count += 1
        
        return decisions
    
    def select_parameters_to_deactivate(self, metrics: WorkloadMetrics) -> List[ActivationDecision]:
        """Select parameters to deactivate based on current metrics"""
        
        decisions = []
        
        # Get candidate parameters for deactivation
        candidates = [
            (pid, self.elastic_parameters[pid]) 
            for pid in self.active_parameters
        ]
        
        # Sort by priority (deactivate lower priority first) and last used
        candidates.sort(key=lambda x: (-x[1].priority, x[1].last_used))
        
        # Calculate how many parameters to deactivate
        max_deactivate = int(len(self.active_parameters) * self.config["max_deactivation_rate"])
        
        deactivated_count = 0
        
        for param_id, param in candidates:
            if deactivated_count >= max_deactivate:
                break
            
            # Check if we should deactivate this parameter
            if self.should_deactivate_parameter(param, metrics):
                decision = ActivationDecision(
                    parameter_id=param_id,
                    action="deactivate",
                    urgency=self.calculate_deactivation_urgency(param, metrics),
                    expected_impact=-param.quality_impact * 0.3,  # Negative impact
                    cost=param.deactivation_cost,
                    reason=f"Load: {metrics.current_load:.2f}, Memory: {metrics.memory_pressure:.2f}"
                )
                decisions.append(decision)
                deactivated_count += 1
        
        return decisions
    
    def can_afford_activation(self, param: ElasticParameter, metrics: WorkloadMetrics) -> bool:
        """Check if we can afford to activate a parameter"""
        
        # Check memory constraints
        current_memory = sum(
            self.elastic_parameters[pid].memory_requirement 
            for pid in self.active_parameters
        )
        hardware_budget = self.hardware_config[param.hardware_target]["memory_budget"]
        available_memory = hardware_budget - current_memory
        
        if param.memory_requirement > available_memory:
            return False
        
        # Check compute constraints
        current_compute = sum(
            self.elastic_parameters[pid].compute_requirement 
            for pid in self.active_parameters
        )
        compute_budget = self.hardware_config[param.hardware_target]["compute_budget"]
        available_compute = compute_budget - current_compute
        
        if param.compute_requirement > available_compute:
            return False
        
        return True
    
    def should_deactivate_parameter(self, param: ElasticParameter, metrics: WorkloadMetrics) -> bool:
        """Check if a parameter should be deactivated"""
        
        # Don't deactivate recently used parameters
        if time.time() - param.last_used < self.config["cooldown_time"]:
            return False
        
        # Don't deactivate high-priority parameters unless absolutely necessary
        if param.priority == 1 and metrics.memory_pressure < 0.95:
            return False
        
        return True
    
    def calculate_activation_urgency(self, param: ElasticParameter, metrics: WorkloadMetrics) -> float:
        """Calculate urgency of parameter activation"""
        
        urgency = 0.0
        
        # Load-based urgency
        if metrics.current_load > 0.8:
            urgency += 0.5
        
        # Response time urgency
        if metrics.response_time > self.config["response_time_target"]:
            urgency += 0.3
        
        # Quality impact urgency
        urgency += param.quality_impact * 0.2
        
        return min(1.0, urgency)
    
    def calculate_deactivation_urgency(self, param: ElasticParameter, metrics: WorkloadMetrics) -> float:
        """Calculate urgency of parameter deactivation"""
        
        urgency = 0.0
        
        # Memory pressure urgency
        if metrics.memory_pressure > 0.9:
            urgency += 0.5
        
        # Low load urgency
        if metrics.current_load < 0.3:
            urgency += 0.3
        
        # Time since last use
        time_since_use = time.time() - param.last_used
        if time_since_use > 60:  # 1 minute
            urgency += 0.2
        
        return min(1.0, urgency)
    
    def execute_decision(self, decision: ActivationDecision):
        """Execute an activation/deactivation decision"""
        
        param_id = decision.parameter_id
        param = self.elastic_parameters[param_id]
        
        if decision.action == "activate":
            self.activate_parameter(param_id, decision.reason)
        elif decision.action == "deactivate":
            self.deactivate_parameter(param_id, decision.reason)
    
    def activate_parameter(self, param_id: str, reason: str = ""):
        """Activate an elastic parameter"""
        
        if param_id not in self.elastic_parameters:
            logger.error(f"Parameter {param_id} not found")
            return False
        
        param = self.elastic_parameters[param_id]
        
        if param.state != ElasticState.INACTIVE:
            logger.warning(f"Parameter {param_id} is not inactive (state: {param.state})")
            return False
        
        try:
            # Update state
            param.state = ElasticState.ACTIVATING
            self.inactive_parameters.discard(param_id)
            self.activating_parameters.add(param_id)
            
            # Simulate activation (in real implementation, this would load the parameter)
            logger.info(f"🔄 Activating parameter {param_id} ({reason})")
            
            # Simulate activation delay
            time.sleep(self.hardware_config[param.hardware_target]["activation_latency"])
            
            # Complete activation
            param.state = ElasticState.ACTIVE
            param.last_used = time.time()
            param.activation_count += 1
            
            self.activating_parameters.discard(param_id)
            self.active_parameters.add(param_id)
            
            # Trigger callbacks
            for callback in self.activation_callbacks:
                try:
                    callback(param_id, param)
                except Exception as e:
                    logger.error(f"Activation callback error: {e}")
            
            logger.info(f"   ✅ Parameter {param_id} activated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to activate parameter {param_id}: {e}")
            
            # Revert state
            param.state = ElasticState.FAILED
            self.activating_parameters.discard(param_id)
            self.inactive_parameters.add(param_id)
            
            return False
    
    def deactivate_parameter(self, param_id: str, reason: str = ""):
        """Deactivate an elastic parameter"""
        
        if param_id not in self.elastic_parameters:
            logger.error(f"Parameter {param_id} not found")
            return False
        
        param = self.elastic_parameters[param_id]
        
        if param.state != ElasticState.ACTIVE:
            logger.warning(f"Parameter {param_id} is not active (state: {param.state})")
            return False
        
        try:
            # Update state
            param.state = ElasticState.DEACTIVATING
            self.active_parameters.discard(param_id)
            self.deactivating_parameters.add(param_id)
            
            # Simulate deactivation (in real implementation, this would unload the parameter)
            logger.info(f"🔄 Deactivating parameter {param_id} ({reason})")
            
            # Simulate deactivation delay
            time.sleep(self.hardware_config[param.hardware_target]["activation_latency"] * 0.5)
            
            # Complete deactivation
            param.state = ElasticState.INACTIVE
            
            self.deactivating_parameters.discard(param_id)
            self.inactive_parameters.add(param_id)
            
            # Trigger callbacks
            for callback in self.deactivation_callbacks:
                try:
                    callback(param_id, param)
                except Exception as e:
                    logger.error(f"Deactivation callback error: {e}")
            
            logger.info(f"   ✅ Parameter {param_id} deactivated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to deactivate parameter {param_id}: {e}")
            
            # Revert state
            param.state = ElasticState.FAILED
            self.deactivating_parameters.discard(param_id)
            self.active_parameters.add(param_id)
            
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        status = {
            "system_active": self.system_active,
            "total_parameters": len(self.elastic_parameters),
            "active_parameters": len(self.active_parameters),
            "inactive_parameters": len(self.inactive_parameters),
            "activating_parameters": len(self.activating_parameters),
            "deactivating_parameters": len(self.deactivating_parameters),
            "memory_usage": {
                hw: sum(
                    self.elastic_parameters[pid].memory_requirement
                    for pid in self.active_parameters
                    if self.elastic_parameters[pid].hardware_target == hw
                ) for hw in self.hardware_config.keys()
            },
            "recent_decisions": len(self.decision_history),
            "recent_metrics": len(self.workload_history)
        }
        
        if self.workload_history:
            latest_metrics = self.workload_history[-1]
            status["current_load"] = latest_metrics.current_load
            status["memory_pressure"] = latest_metrics.memory_pressure
            status["response_time"] = latest_metrics.response_time
            status["throughput"] = latest_metrics.throughput
        
        return status
    
    def add_activation_callback(self, callback: Callable):
        """Add callback for parameter activation"""
        self.activation_callbacks.append(callback)
    
    def add_deactivation_callback(self, callback: Callable):
        """Add callback for parameter deactivation"""
        self.deactivation_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable):
        """Add callback for metrics updates"""
        self.metrics_callbacks.append(callback)
    
    def save_system_state(self, output_path: str):
        """Save current system state"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving elastic activation system state to {output_dir}")
        
        # Save system status
        status_file = output_dir / "system_status.json"
        with open(status_file, 'w') as f:
            json.dump(self.get_system_status(), f, indent=2)
        
        # Save configuration
        config_file = output_dir / "system_config.json"
        with open(config_file, 'w') as f:
            config_data = {
                "config": self.config,
                "hardware_config": self.hardware_config,
                "timestamp": time.time()
            }
            json.dump(config_data, f, indent=2)
        
        # Save parameter states
        params_file = output_dir / "parameter_states.json"
        with open(params_file, 'w') as f:
            params_data = {
                pid: {
                    "layer_id": param.layer_id,
                    "parameter_type": param.parameter_type,
                    "memory_requirement": param.memory_requirement,
                    "compute_requirement": param.compute_requirement,
                    "priority": param.priority,
                    "state": param.state.value,
                    "hardware_target": param.hardware_target,
                    "last_used": param.last_used,
                    "activation_count": param.activation_count
                } for pid, param in self.elastic_parameters.items()
            }
            json.dump(params_data, f, indent=2)
        
        logger.info("✅ System state saved successfully!")
        
        return output_dir

def main():
    """Main function for testing the elastic activation system"""
    
    logger.info("🦄 Gemma 3n E4B Elastic Parameter Activation System")
    logger.info("=" * 60)
    
    # Initialize system
    system = Gemma3nE4BElasticActivationSystem()
    
    # Add some test callbacks
    def activation_callback(param_id: str, param: ElasticParameter):
        logger.info(f"🔵 Callback: Parameter {param_id} activated on {param.hardware_target}")
    
    def deactivation_callback(param_id: str, param: ElasticParameter):
        logger.info(f"🔴 Callback: Parameter {param_id} deactivated from {param.hardware_target}")
    
    def metrics_callback(metrics: WorkloadMetrics):
        if int(metrics.timestamp) % 10 == 0:  # Log every 10 seconds
            logger.info(f"📊 Load: {metrics.current_load:.2f}, RT: {metrics.response_time:.1f}ms, "
                       f"TP: {metrics.throughput:.1f} TPS")
    
    system.add_activation_callback(activation_callback)
    system.add_deactivation_callback(deactivation_callback)
    system.add_metrics_callback(metrics_callback)
    
    # Start monitoring
    system.start_monitoring()
    
    try:
        # Run for 30 seconds
        logger.info("🚀 Running elastic activation system for 30 seconds...")
        time.sleep(30)
        
        # Print final status
        status = system.get_system_status()
        logger.info("📊 Final System Status:")
        logger.info(f"   Active parameters: {status['active_parameters']}")
        logger.info(f"   Inactive parameters: {status['inactive_parameters']}")
        logger.info(f"   Current load: {status.get('current_load', 0):.2f}")
        logger.info(f"   Memory pressure: {status.get('memory_pressure', 0):.2f}")
        
        # Save system state
        output_path = "./elastic_system_states/gemma-3n-e4b-test-run"
        system.save_system_state(output_path)
        
    except KeyboardInterrupt:
        logger.info("🛑 Stopping system...")
    
    finally:
        system.stop_monitoring()
    
    logger.info("=" * 60)
    logger.info("🎯 ELASTIC ACTIVATION SYSTEM TEST COMPLETE!")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())