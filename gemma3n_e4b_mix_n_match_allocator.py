#!/usr/bin/env python3
"""
Gemma 3n E4B Mix-n-Match Layer Allocation Strategy
Dynamic layer allocation for NPU+iGPU with elastic parameter support
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LayerAllocation:
    """Represents a layer allocation to specific hardware"""
    layer_id: int
    layer_type: str
    hardware_target: str
    memory_requirement: int
    compute_requirement: float
    elastic_support: bool
    priority: int

@dataclass
class ElasticConfiguration:
    """Configuration for elastic parameter management"""
    base_active_layers: int
    elastic_active_layers: int
    activation_threshold: float
    deactivation_threshold: float
    scaling_factor: float

class Gemma3nE4BMixNMatchAllocator:
    """Mix-n-Match layer allocation strategy for Gemma 3n E4B elastic parameters"""
    
    def __init__(self, model_path: str = "./models/gemma-3n-e4b-it"):
        self.model_path = Path(model_path)
        
        # Hardware configuration
        self.hardware_config = {
            "npu_phoenix": {
                "memory_budget": 2 * 1024**3,  # 2GB NPU SRAM
                "compute_budget": 16 * 1024**3,  # 16 TOPS
                "precision": "INT8",
                "preferred_ops": ["attention", "embedding", "layer_norm"],
                "max_layers": 16,  # Maximum layers NPU can handle
                "turbo_mode": True,
                "latency_priority": True
            },
            "radeon_780m": {
                "memory_budget": 16 * 1024**3,  # 16GB DDR5 allocation
                "compute_budget": 2.7 * 1024**3,  # 2.7 TFLOPS
                "precision": "INT4",
                "preferred_ops": ["ffn", "output_projection", "matrix_multiply"],
                "max_layers": 32,  # Maximum layers iGPU can handle
                "parallel_streams": 4,
                "bandwidth_priority": True
            },
            "system_memory": {
                "memory_budget": 80 * 1024**3,  # 80GB DDR5 available
                "compute_budget": 1.2 * 1024**3,  # 1.2 TFLOPS (CPU)
                "precision": "FP16",
                "preferred_ops": ["inactive_params", "orchestration", "fallback"],
                "max_layers": 128,  # Unlimited layers
                "storage_priority": True
            }
        }
        
        # Mix-n-Match configuration
        self.mix_n_match_config = {
            "target_performance": 200,  # Target TPS
            "memory_efficiency": 0.85,  # 85% memory utilization target
            "compute_efficiency": 0.90,  # 90% compute utilization target
            "elastic_scaling_enabled": True,
            "dynamic_reallocation": True,
            "load_balancing": True,
            "quality_preservation": 0.95  # 95% quality preservation
        }
        
        # Elastic parameter configuration
        self.elastic_config = ElasticConfiguration(
            base_active_layers=16,  # Always active layers
            elastic_active_layers=8,  # Can be activated dynamically
            activation_threshold=0.7,  # Activate when load > 70%
            deactivation_threshold=0.3,  # Deactivate when load < 30%
            scaling_factor=1.5  # Performance scaling factor
        )
        
        # Current allocation state
        self.current_allocation = {}
        self.performance_history = []
        self.load_history = []
        
    def analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze Gemma 3n E4B architecture for optimal allocation"""
        logger.info("🔍 Analyzing Gemma 3n E4B architecture...")
        
        # Gemma 3n E4B architecture (from model card and technical specs)
        architecture = {
            "model_type": "gemma3n-e4b",
            "base_params": 2 * 1024**3,  # 2B effective parameters
            "total_params": 4 * 1024**3,  # 4B total parameters
            "elastic_params": 2 * 1024**3,  # 2B elastic parameters
            "num_layers": 24,  # Estimated based on E4B configuration
            "hidden_size": 3072,  # Estimated
            "num_attention_heads": 24,  # Estimated
            "num_key_value_heads": 8,  # GQA configuration
            "intermediate_size": 8192,  # FFN intermediate size
            "context_length": 32768,  # 32K context support
            "vocab_size": 256000,  # Estimated vocabulary size
            "attention_dropout": 0.1,
            "hidden_dropout": 0.1,
            "activation_function": "gelu",
            "layer_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "elastic_ratio": 0.5,  # 50% elastic parameters
            "mix_n_match_enabled": True
        }
        
        # Calculate layer-wise memory and compute requirements
        layers_info = self.calculate_layer_requirements(architecture)
        
        # Analyze elastic parameter patterns
        elastic_patterns = self.analyze_elastic_patterns(architecture)
        
        # Create allocation matrix
        allocation_matrix = self.create_allocation_matrix(architecture, layers_info)
        
        logger.info(f"   📊 Model: {architecture['model_type']}")
        logger.info(f"   📊 Layers: {architecture['num_layers']}")
        logger.info(f"   📊 Base params: {architecture['base_params']/1e9:.1f}B")
        logger.info(f"   📊 Elastic params: {architecture['elastic_params']/1e9:.1f}B")
        logger.info(f"   📊 Context length: {architecture['context_length']:,}")
        
        return {
            "architecture": architecture,
            "layers_info": layers_info,
            "elastic_patterns": elastic_patterns,
            "allocation_matrix": allocation_matrix
        }
    
    def calculate_layer_requirements(self, arch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate memory and compute requirements for each layer"""
        
        hidden_size = arch["hidden_size"]
        intermediate_size = arch["intermediate_size"]
        num_heads = arch["num_attention_heads"]
        num_layers = arch["num_layers"]
        
        layers_info = []
        
        for layer_idx in range(num_layers):
            # Attention layer requirements
            attention_memory = (
                4 * hidden_size * hidden_size +  # Q, K, V, O projections
                2 * num_heads * (hidden_size // num_heads) * 32768  # KV cache
            )
            
            attention_compute = (
                3 * hidden_size * hidden_size +  # QKV computation
                2 * 32768 * hidden_size  # Attention computation
            )
            
            # FFN layer requirements
            ffn_memory = (
                hidden_size * intermediate_size +  # Gate projection
                hidden_size * intermediate_size +  # Up projection
                intermediate_size * hidden_size    # Down projection
            )
            
            ffn_compute = (
                2 * hidden_size * intermediate_size +  # Gate + Up
                intermediate_size * hidden_size        # Down
            )
            
            # Layer norm requirements
            layernorm_memory = hidden_size * 2  # Weight + bias
            layernorm_compute = hidden_size * 32768  # Normalization
            
            layer_info = {
                "layer_id": layer_idx,
                "attention": {
                    "memory": attention_memory,
                    "compute": attention_compute,
                    "type": "attention",
                    "elastic_support": True
                },
                "ffn": {
                    "memory": ffn_memory,
                    "compute": ffn_compute,
                    "type": "ffn",
                    "elastic_support": True
                },
                "layernorm": {
                    "memory": layernorm_memory,
                    "compute": layernorm_compute,
                    "type": "layernorm",
                    "elastic_support": False
                },
                "total_memory": attention_memory + ffn_memory + layernorm_memory,
                "total_compute": attention_compute + ffn_compute + layernorm_compute,
                "elastic_memory": (attention_memory + ffn_memory) * 0.5,  # 50% elastic
                "priority": self.calculate_layer_priority(layer_idx, num_layers)
            }
            
            layers_info.append(layer_info)
        
        return layers_info
    
    def calculate_layer_priority(self, layer_idx: int, num_layers: int) -> int:
        """Calculate priority for layer allocation"""
        
        # Early layers (embedding-like) get highest priority
        if layer_idx < 4:
            return 1
        
        # Middle layers (main processing) get medium priority
        elif layer_idx < num_layers - 4:
            return 2
        
        # Late layers (output processing) get high priority
        else:
            return 1
    
    def analyze_elastic_patterns(self, arch: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze elastic parameter activation patterns"""
        
        num_layers = arch["num_layers"]
        elastic_ratio = arch["elastic_ratio"]
        
        # Calculate elastic activation patterns
        patterns = {
            "base_layers": int(num_layers * (1 - elastic_ratio)),
            "elastic_layers": int(num_layers * elastic_ratio),
            "activation_order": self.calculate_activation_order(num_layers),
            "deactivation_order": self.calculate_deactivation_order(num_layers),
            "scaling_factors": self.calculate_scaling_factors(num_layers),
            "load_thresholds": self.calculate_load_thresholds(num_layers)
        }
        
        return patterns
    
    def calculate_activation_order(self, num_layers: int) -> List[int]:
        """Calculate optimal order for activating elastic parameters"""
        
        # Activate middle layers first (most compute-intensive)
        middle_start = num_layers // 4
        middle_end = 3 * num_layers // 4
        
        activation_order = []
        
        # Start with middle layers
        for i in range(middle_start, middle_end):
            activation_order.append(i)
        
        # Then early layers
        for i in range(middle_start):
            activation_order.append(i)
        
        # Finally late layers
        for i in range(middle_end, num_layers):
            activation_order.append(i)
        
        return activation_order
    
    def calculate_deactivation_order(self, num_layers: int) -> List[int]:
        """Calculate optimal order for deactivating elastic parameters"""
        
        # Deactivate in reverse order of activation
        activation_order = self.calculate_activation_order(num_layers)
        return activation_order[::-1]
    
    def calculate_scaling_factors(self, num_layers: int) -> List[float]:
        """Calculate scaling factors for each layer"""
        
        scaling_factors = []
        
        for layer_idx in range(num_layers):
            # Higher scaling for middle layers
            if layer_idx < num_layers // 4:
                factor = 1.2
            elif layer_idx < 3 * num_layers // 4:
                factor = 1.5
            else:
                factor = 1.3
            
            scaling_factors.append(factor)
        
        return scaling_factors
    
    def calculate_load_thresholds(self, num_layers: int) -> List[float]:
        """Calculate load thresholds for elastic activation"""
        
        thresholds = []
        
        for layer_idx in range(num_layers):
            # Lower threshold for critical layers
            if layer_idx < 4 or layer_idx >= num_layers - 4:
                threshold = 0.6
            else:
                threshold = 0.7
            
            thresholds.append(threshold)
        
        return thresholds
    
    def create_allocation_matrix(self, arch: Dict[str, Any], layers_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create optimal allocation matrix for Mix-n-Match strategy"""
        
        num_layers = arch["num_layers"]
        
        # Initialize allocation matrix
        allocation_matrix = {
            "npu_phoenix": {
                "base_layers": [],
                "elastic_layers": [],
                "memory_used": 0,
                "compute_used": 0,
                "utilization": 0.0
            },
            "radeon_780m": {
                "base_layers": [],
                "elastic_layers": [],
                "memory_used": 0,
                "compute_used": 0,
                "utilization": 0.0
            },
            "system_memory": {
                "base_layers": [],
                "elastic_layers": [],
                "memory_used": 0,
                "compute_used": 0,
                "utilization": 0.0
            }
        }
        
        # Allocate layers using Mix-n-Match strategy
        allocation_matrix = self.allocate_layers_optimally(
            layers_info, allocation_matrix, arch
        )
        
        # Calculate utilization metrics
        allocation_matrix = self.calculate_utilization_metrics(allocation_matrix)
        
        return allocation_matrix
    
    def allocate_layers_optimally(self, layers_info: List[Dict[str, Any]], 
                                 allocation_matrix: Dict[str, Any], 
                                 arch: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate layers optimally across hardware using Mix-n-Match"""
        
        # Sort layers by priority and compute requirements
        sorted_layers = sorted(layers_info, key=lambda x: (x["priority"], -x["total_compute"]))
        
        for layer_info in sorted_layers:
            layer_id = layer_info["layer_id"]
            
            # Allocate attention operations
            attention_target = self.select_hardware_for_attention(layer_info, allocation_matrix)
            self.allocate_component(layer_info["attention"], attention_target, allocation_matrix, layer_id, "attention")
            
            # Allocate FFN operations
            ffn_target = self.select_hardware_for_ffn(layer_info, allocation_matrix)
            self.allocate_component(layer_info["ffn"], ffn_target, allocation_matrix, layer_id, "ffn")
            
            # Allocate layer norm operations
            layernorm_target = self.select_hardware_for_layernorm(layer_info, allocation_matrix)
            self.allocate_component(layer_info["layernorm"], layernorm_target, allocation_matrix, layer_id, "layernorm")
        
        return allocation_matrix
    
    def select_hardware_for_attention(self, layer_info: Dict[str, Any], 
                                    allocation_matrix: Dict[str, Any]) -> str:
        """Select optimal hardware for attention operations"""
        
        attention_memory = layer_info["attention"]["memory"]
        attention_compute = layer_info["attention"]["compute"]
        
        # NPU Phoenix is preferred for attention operations
        npu_memory_available = (
            self.hardware_config["npu_phoenix"]["memory_budget"] - 
            allocation_matrix["npu_phoenix"]["memory_used"]
        )
        
        if npu_memory_available >= attention_memory:
            return "npu_phoenix"
        
        # Fallback to iGPU if NPU is full
        igpu_memory_available = (
            self.hardware_config["radeon_780m"]["memory_budget"] - 
            allocation_matrix["radeon_780m"]["memory_used"]
        )
        
        if igpu_memory_available >= attention_memory:
            return "radeon_780m"
        
        # Final fallback to system memory
        return "system_memory"
    
    def select_hardware_for_ffn(self, layer_info: Dict[str, Any], 
                               allocation_matrix: Dict[str, Any]) -> str:
        """Select optimal hardware for FFN operations"""
        
        ffn_memory = layer_info["ffn"]["memory"]
        ffn_compute = layer_info["ffn"]["compute"]
        
        # iGPU is preferred for FFN operations
        igpu_memory_available = (
            self.hardware_config["radeon_780m"]["memory_budget"] - 
            allocation_matrix["radeon_780m"]["memory_used"]
        )
        
        if igpu_memory_available >= ffn_memory:
            return "radeon_780m"
        
        # Fallback to NPU if iGPU is full
        npu_memory_available = (
            self.hardware_config["npu_phoenix"]["memory_budget"] - 
            allocation_matrix["npu_phoenix"]["memory_used"]
        )
        
        if npu_memory_available >= ffn_memory:
            return "npu_phoenix"
        
        # Final fallback to system memory
        return "system_memory"
    
    def select_hardware_for_layernorm(self, layer_info: Dict[str, Any], 
                                    allocation_matrix: Dict[str, Any]) -> str:
        """Select optimal hardware for layer normalization"""
        
        # Layer norm is lightweight, prefer NPU for latency
        layernorm_memory = layer_info["layernorm"]["memory"]
        
        npu_memory_available = (
            self.hardware_config["npu_phoenix"]["memory_budget"] - 
            allocation_matrix["npu_phoenix"]["memory_used"]
        )
        
        if npu_memory_available >= layernorm_memory:
            return "npu_phoenix"
        
        # Otherwise use system memory (CPU is efficient for layer norm)
        return "system_memory"
    
    def allocate_component(self, component_info: Dict[str, Any], hardware_target: str,
                          allocation_matrix: Dict[str, Any], layer_id: int, 
                          component_type: str):
        """Allocate a component to specific hardware"""
        
        # Create allocation record
        allocation = LayerAllocation(
            layer_id=layer_id,
            layer_type=component_type,
            hardware_target=hardware_target,
            memory_requirement=component_info["memory"],
            compute_requirement=component_info["compute"],
            elastic_support=component_info["elastic_support"],
            priority=1 if component_type == "attention" else 2
        )
        
        # Update allocation matrix
        allocation_matrix[hardware_target]["base_layers"].append(allocation)
        allocation_matrix[hardware_target]["memory_used"] += component_info["memory"]
        allocation_matrix[hardware_target]["compute_used"] += component_info["compute"]
        
        # Add to elastic layers if supported
        if component_info["elastic_support"]:
            allocation_matrix[hardware_target]["elastic_layers"].append(allocation)
    
    def calculate_utilization_metrics(self, allocation_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate utilization metrics for each hardware component"""
        
        for hardware_name, hardware_info in allocation_matrix.items():
            hardware_config = self.hardware_config[hardware_name]
            
            # Calculate memory utilization
            memory_utilization = (
                hardware_info["memory_used"] / hardware_config["memory_budget"]
            )
            
            # Calculate compute utilization
            compute_utilization = (
                hardware_info["compute_used"] / hardware_config["compute_budget"]
            )
            
            # Overall utilization
            overall_utilization = (memory_utilization + compute_utilization) / 2
            
            allocation_matrix[hardware_name]["utilization"] = overall_utilization
            allocation_matrix[hardware_name]["memory_utilization"] = memory_utilization
            allocation_matrix[hardware_name]["compute_utilization"] = compute_utilization
        
        return allocation_matrix
    
    def optimize_allocation(self, allocation_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize allocation using Mix-n-Match strategy"""
        
        logger.info("🔧 Optimizing Mix-n-Match allocation...")
        
        # Analyze current allocation efficiency
        efficiency_metrics = self.analyze_allocation_efficiency(allocation_matrix)
        
        # Rebalance if needed
        if efficiency_metrics["needs_rebalancing"]:
            allocation_matrix = self.rebalance_allocation(allocation_matrix)
        
        # Optimize elastic parameter allocation
        allocation_matrix = self.optimize_elastic_allocation(allocation_matrix)
        
        # Final validation
        validation_results = self.validate_allocation(allocation_matrix)
        
        logger.info(f"   ✅ NPU utilization: {allocation_matrix['npu_phoenix']['utilization']:.1%}")
        logger.info(f"   ✅ iGPU utilization: {allocation_matrix['radeon_780m']['utilization']:.1%}")
        logger.info(f"   ✅ System utilization: {allocation_matrix['system_memory']['utilization']:.1%}")
        
        return allocation_matrix
    
    def analyze_allocation_efficiency(self, allocation_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze efficiency of current allocation"""
        
        # Calculate efficiency metrics
        npu_util = allocation_matrix["npu_phoenix"]["utilization"]
        igpu_util = allocation_matrix["radeon_780m"]["utilization"]
        system_util = allocation_matrix["system_memory"]["utilization"]
        
        # Check for imbalances
        utilization_variance = np.var([npu_util, igpu_util, system_util])
        needs_rebalancing = utilization_variance > 0.1  # 10% variance threshold
        
        # Calculate overall efficiency
        overall_efficiency = (npu_util + igpu_util + system_util) / 3
        
        return {
            "overall_efficiency": overall_efficiency,
            "utilization_variance": utilization_variance,
            "needs_rebalancing": needs_rebalancing,
            "npu_util": npu_util,
            "igpu_util": igpu_util,
            "system_util": system_util
        }
    
    def rebalance_allocation(self, allocation_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Rebalance allocation across hardware"""
        
        logger.info("   🔄 Rebalancing allocation...")
        
        # Find overloaded and underloaded hardware
        hardware_loads = [
            ("npu_phoenix", allocation_matrix["npu_phoenix"]["utilization"]),
            ("radeon_780m", allocation_matrix["radeon_780m"]["utilization"]),
            ("system_memory", allocation_matrix["system_memory"]["utilization"])
        ]
        
        hardware_loads.sort(key=lambda x: x[1])
        
        # Move layers from overloaded to underloaded hardware
        for i in range(len(hardware_loads) - 1):
            underloaded_hw = hardware_loads[i][0]
            overloaded_hw = hardware_loads[i + 1][0]
            
            if hardware_loads[i + 1][1] - hardware_loads[i][1] > 0.2:  # 20% difference
                allocation_matrix = self.move_layers(allocation_matrix, overloaded_hw, underloaded_hw)
        
        return allocation_matrix
    
    def move_layers(self, allocation_matrix: Dict[str, Any], source_hw: str, target_hw: str) -> Dict[str, Any]:
        """Move layers from source to target hardware"""
        
        source_layers = allocation_matrix[source_hw]["base_layers"]
        target_budget = self.hardware_config[target_hw]["memory_budget"]
        target_used = allocation_matrix[target_hw]["memory_used"]
        
        # Sort layers by priority and memory usage
        movable_layers = sorted(source_layers, key=lambda x: (x.priority, -x.memory_requirement))
        
        for layer in movable_layers:
            if target_used + layer.memory_requirement <= target_budget:
                # Move layer
                allocation_matrix[source_hw]["base_layers"].remove(layer)
                allocation_matrix[source_hw]["memory_used"] -= layer.memory_requirement
                allocation_matrix[source_hw]["compute_used"] -= layer.compute_requirement
                
                # Update layer target
                layer.hardware_target = target_hw
                
                allocation_matrix[target_hw]["base_layers"].append(layer)
                allocation_matrix[target_hw]["memory_used"] += layer.memory_requirement
                allocation_matrix[target_hw]["compute_used"] += layer.compute_requirement
                
                target_used += layer.memory_requirement
                
                # Stop if we've balanced enough
                if abs(allocation_matrix[source_hw]["utilization"] - allocation_matrix[target_hw]["utilization"]) < 0.1:
                    break
        
        return allocation_matrix
    
    def optimize_elastic_allocation(self, allocation_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize elastic parameter allocation"""
        
        logger.info("   🔄 Optimizing elastic parameter allocation...")
        
        for hardware_name, hardware_info in allocation_matrix.items():
            elastic_layers = hardware_info["elastic_layers"]
            
            # Sort elastic layers by activation priority
            elastic_layers.sort(key=lambda x: x.priority)
            
            # Update elastic configuration
            hardware_info["elastic_config"] = {
                "max_elastic_layers": len(elastic_layers),
                "current_active_layers": len(elastic_layers) // 2,  # Start with 50% active
                "activation_threshold": 0.7,
                "deactivation_threshold": 0.3,
                "scaling_enabled": True
            }
        
        return allocation_matrix
    
    def validate_allocation(self, allocation_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the final allocation"""
        
        validation_results = {
            "valid": True,
            "issues": [],
            "metrics": {}
        }
        
        for hardware_name, hardware_info in allocation_matrix.items():
            hardware_config = self.hardware_config[hardware_name]
            
            # Check memory constraints
            if hardware_info["memory_used"] > hardware_config["memory_budget"]:
                validation_results["valid"] = False
                validation_results["issues"].append(f"{hardware_name}: Memory budget exceeded")
            
            # Check compute constraints
            if hardware_info["compute_used"] > hardware_config["compute_budget"]:
                validation_results["valid"] = False
                validation_results["issues"].append(f"{hardware_name}: Compute budget exceeded")
            
            # Calculate efficiency metrics
            validation_results["metrics"][hardware_name] = {
                "memory_utilization": hardware_info["memory_used"] / hardware_config["memory_budget"],
                "compute_utilization": hardware_info["compute_used"] / hardware_config["compute_budget"],
                "overall_utilization": hardware_info["utilization"]
            }
        
        return validation_results
    
    def save_allocation_strategy(self, allocation_matrix: Dict[str, Any], output_path: str):
        """Save the Mix-n-Match allocation strategy"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving Mix-n-Match allocation strategy to {output_dir}")
        
        # Save allocation matrix
        allocation_file = output_dir / "mix_n_match_allocation.json"
        with open(allocation_file, 'w') as f:
            # Convert LayerAllocation objects to dict for JSON serialization
            json_allocation = {}
            for hw_name, hw_info in allocation_matrix.items():
                json_allocation[hw_name] = {
                    "base_layers": [
                        {
                            "layer_id": layer.layer_id,
                            "layer_type": layer.layer_type,
                            "hardware_target": layer.hardware_target,
                            "memory_requirement": layer.memory_requirement,
                            "compute_requirement": layer.compute_requirement,
                            "elastic_support": layer.elastic_support,
                            "priority": layer.priority
                        } for layer in hw_info["base_layers"]
                    ],
                    "elastic_layers": [
                        {
                            "layer_id": layer.layer_id,
                            "layer_type": layer.layer_type,
                            "hardware_target": layer.hardware_target,
                            "memory_requirement": layer.memory_requirement,
                            "compute_requirement": layer.compute_requirement,
                            "elastic_support": layer.elastic_support,
                            "priority": layer.priority
                        } for layer in hw_info["elastic_layers"]
                    ],
                    "memory_used": hw_info["memory_used"],
                    "compute_used": hw_info["compute_used"],
                    "utilization": hw_info["utilization"]
                }
            
            json.dump(json_allocation, f, indent=2)
        
        # Save configuration
        config_file = output_dir / "mix_n_match_config.json"
        with open(config_file, 'w') as f:
            config_data = {
                "hardware_config": self.hardware_config,
                "mix_n_match_config": self.mix_n_match_config,
                "elastic_config": {
                    "base_active_layers": self.elastic_config.base_active_layers,
                    "elastic_active_layers": self.elastic_config.elastic_active_layers,
                    "activation_threshold": self.elastic_config.activation_threshold,
                    "deactivation_threshold": self.elastic_config.deactivation_threshold,
                    "scaling_factor": self.elastic_config.scaling_factor
                },
                "timestamp": time.time()
            }
            json.dump(config_data, f, indent=2)
        
        logger.info("✅ Mix-n-Match allocation strategy saved successfully!")
        
        return output_dir

def main():
    """Main allocation function"""
    
    logger.info("🦄 Gemma 3n E4B Mix-n-Match Layer Allocator")
    logger.info("=" * 60)
    
    # Initialize allocator
    allocator = Gemma3nE4BMixNMatchAllocator()
    
    # Analyze model architecture
    start_time = time.time()
    model_info = allocator.analyze_model_architecture()
    
    if not model_info:
        logger.error("❌ Model architecture analysis failed")
        return 1
    
    # Get allocation matrix
    allocation_matrix = model_info["allocation_matrix"]
    
    # Optimize allocation
    logger.info("🚀 Optimizing Mix-n-Match allocation...")
    optimized_allocation = allocator.optimize_allocation(allocation_matrix)
    
    # Save allocation strategy
    output_path = "./allocation_strategies/gemma-3n-e4b-mix-n-match"
    allocator.save_allocation_strategy(optimized_allocation, output_path)
    
    # Performance summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🎯 MIX-N-MATCH ALLOCATION COMPLETE!")
    logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
    logger.info(f"📁 Output: {output_path}")
    logger.info(f"🔧 Hardware: NPU Phoenix + Radeon 780M + Mix-n-Match")
    logger.info(f"📊 Elastic scaling: Enabled with dynamic allocation")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())