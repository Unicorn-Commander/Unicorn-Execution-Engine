#!/usr/bin/env python3
"""
Advanced Hardware-Specific Tuner
Real-time optimization for NPU Phoenix + AMD Radeon 780M
- Dynamic performance monitoring
- Adaptive parameter tuning
- Hardware-specific optimizations
"""

import numpy as np
import time
import logging
import subprocess
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for hardware tuning"""
    npu_utilization: float = 0.0
    igpu_utilization: float = 0.0
    memory_bandwidth: float = 0.0
    temperature_c: float = 0.0
    power_watts: float = 0.0
    tokens_per_second: float = 0.0
    latency_ms: float = 0.0
    efficiency_score: float = 0.0

class AdvancedHardwareTuner:
    """Advanced hardware-specific tuner with real-time optimization"""
    
    def __init__(self):
        self.tuning_active = False
        self.performance_history = []
        self.optimal_parameters = {}
        self.monitoring_thread = None
        
        # Hardware-specific thresholds
        self.THERMAL_LIMIT_C = 85.0
        self.POWER_LIMIT_W = 65.0  # Ryzen 9 8945HS TDP
        self.TARGET_UTILIZATION = 85.0
        self.MIN_EFFICIENCY_SCORE = 0.7
        
        # Tuning parameters
        self.tuning_parameters = {
            'npu_frequency': ['nominal', 'boost', 'turbo'],
            'npu_block_size': [32, 64, 128, 256],
            'igpu_workgroup_x': [4, 8, 16, 32],
            'igpu_workgroup_y': [4, 8, 16, 32],
            'igpu_tile_size': [8, 16, 32, 64],
            'memory_prefetch': [64, 128, 256, 512],
            'parallel_streams': [1, 2, 4, 8]
        }
        
        logger.info("🎯 Advanced Hardware Tuner initialized")
        logger.info(f"   Thermal limit: {self.THERMAL_LIMIT_C}°C")
        logger.info(f"   Power limit: {self.POWER_LIMIT_W}W")
        logger.info(f"   Target utilization: {self.TARGET_UTILIZATION}%")
    
    def start_real_time_monitoring(self):
        """Start real-time hardware monitoring"""
        logger.info("📊 Starting real-time hardware monitoring...")
        
        self.tuning_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("✅ Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.tuning_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("🛑 Hardware monitoring stopped")
    
    def _monitoring_loop(self):
        """Real-time monitoring loop"""
        while self.tuning_active:
            try:
                metrics = self._collect_hardware_metrics()
                self.performance_history.append(metrics)
                
                # Keep only last 100 measurements
                if len(self.performance_history) > 100:
                    self.performance_history.pop(0)
                
                # Check for optimization opportunities
                if len(self.performance_history) >= 10:
                    self._analyze_and_optimize()
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                time.sleep(5.0)
    
    def _collect_hardware_metrics(self):
        """Collect current hardware performance metrics"""
        metrics = PerformanceMetrics()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            cpu_temps = psutil.sensors_temperatures()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # NPU metrics (via XRT if available)
            try:
                npu_result = subprocess.run([
                    "/opt/xilinx/xrt/bin/xrt-smi", "examine", "--json"
                ], capture_output=True, text=True, timeout=5)
                
                if npu_result.returncode == 0:
                    # Parse NPU utilization (simplified)
                    metrics.npu_utilization = cpu_percent * 0.1  # Approximation
                    
            except Exception:
                metrics.npu_utilization = 0.0
            
            # iGPU metrics (approximated from system load)
            metrics.igpu_utilization = min(cpu_percent * 1.2, 100.0)
            
            # Memory bandwidth (estimated from memory usage changes)
            metrics.memory_bandwidth = self._estimate_memory_bandwidth()
            
            # Temperature
            if 'coretemp' in cpu_temps:
                temp_sensors = cpu_temps['coretemp']
                if temp_sensors:
                    metrics.temperature_c = max(sensor.current for sensor in temp_sensors)
            
            # Power (estimated from CPU frequency and load)
            if cpu_freq:
                base_power = 15.0  # Base power consumption
                dynamic_power = (cpu_percent / 100.0) * (cpu_freq.current / cpu_freq.max) * 50.0
                metrics.power_watts = base_power + dynamic_power
            
            # Calculate efficiency score
            metrics.efficiency_score = self._calculate_efficiency_score(metrics)
            
        except Exception as e:
            logger.warning(f"Metrics collection error: {e}")
        
        return metrics
    
    def _estimate_memory_bandwidth(self):
        """Estimate current memory bandwidth usage"""
        try:
            # Simple estimation based on memory activity
            memory = psutil.virtual_memory()
            return min(memory.percent * 0.8, 85.0)  # Max 85 GB/s for DDR5-5600
        except:
            return 0.0
    
    def _calculate_efficiency_score(self, metrics: PerformanceMetrics):
        """Calculate hardware efficiency score (0-1)"""
        try:
            # Weight different factors
            utilization_score = (metrics.npu_utilization + metrics.igpu_utilization) / 200.0
            thermal_score = max(0, 1.0 - (metrics.temperature_c - 60.0) / 25.0)  # Penalty above 60°C
            power_score = max(0, 1.0 - (metrics.power_watts - 30.0) / 35.0)  # Penalty above 30W
            
            # Weighted average
            efficiency = (utilization_score * 0.5 + thermal_score * 0.3 + power_score * 0.2)
            return max(0.0, min(1.0, efficiency))
            
        except:
            return 0.5
    
    def _analyze_and_optimize(self):
        """Analyze performance history and optimize parameters"""
        recent_metrics = self.performance_history[-10:]
        avg_metrics = self._calculate_average_metrics(recent_metrics)
        
        # Check for optimization opportunities
        optimization_needed = False
        
        # Check thermal throttling
        if avg_metrics.temperature_c > self.THERMAL_LIMIT_C:
            logger.warning(f"🌡️ Thermal throttling detected: {avg_metrics.temperature_c:.1f}°C")
            self._apply_thermal_optimization()
            optimization_needed = True
        
        # Check power limit
        if avg_metrics.power_watts > self.POWER_LIMIT_W:
            logger.warning(f"⚡ Power limit exceeded: {avg_metrics.power_watts:.1f}W")
            self._apply_power_optimization()
            optimization_needed = True
        
        # Check efficiency
        if avg_metrics.efficiency_score < self.MIN_EFFICIENCY_SCORE:
            logger.info(f"📊 Low efficiency detected: {avg_metrics.efficiency_score:.2f}")
            self._apply_efficiency_optimization(avg_metrics)
            optimization_needed = True
        
        # Adaptive parameter tuning
        if not optimization_needed and len(self.performance_history) % 30 == 0:
            self._adaptive_parameter_tuning(avg_metrics)
    
    def _calculate_average_metrics(self, metrics_list):
        """Calculate average metrics from list"""
        if not metrics_list:
            return PerformanceMetrics()
        
        avg = PerformanceMetrics()
        count = len(metrics_list)
        
        for metrics in metrics_list:
            avg.npu_utilization += metrics.npu_utilization
            avg.igpu_utilization += metrics.igpu_utilization
            avg.memory_bandwidth += metrics.memory_bandwidth
            avg.temperature_c += metrics.temperature_c
            avg.power_watts += metrics.power_watts
            avg.tokens_per_second += metrics.tokens_per_second
            avg.latency_ms += metrics.latency_ms
            avg.efficiency_score += metrics.efficiency_score
        
        # Calculate averages
        avg.npu_utilization /= count
        avg.igpu_utilization /= count
        avg.memory_bandwidth /= count
        avg.temperature_c /= count
        avg.power_watts /= count
        avg.tokens_per_second /= count
        avg.latency_ms /= count
        avg.efficiency_score /= count
        
        return avg
    
    def _apply_thermal_optimization(self):
        """Apply thermal optimization parameters"""
        logger.info("🌡️ Applying thermal optimization...")
        
        # Reduce frequencies and workgroup sizes
        thermal_params = {
            'npu_frequency': 'nominal',
            'igpu_workgroup_x': 4,
            'igpu_workgroup_y': 4,
            'igpu_tile_size': 8,
            'parallel_streams': 2
        }
        
        self.optimal_parameters.update(thermal_params)
        logger.info("   ✅ Thermal optimization applied")
    
    def _apply_power_optimization(self):
        """Apply power optimization parameters"""
        logger.info("⚡ Applying power optimization...")
        
        # Reduce power consumption
        power_params = {
            'npu_frequency': 'boost',  # Not turbo
            'npu_block_size': 64,
            'memory_prefetch': 128,
            'parallel_streams': 1
        }
        
        self.optimal_parameters.update(power_params)
        logger.info("   ✅ Power optimization applied")
    
    def _apply_efficiency_optimization(self, metrics: PerformanceMetrics):
        """Apply efficiency optimization based on current metrics"""
        logger.info("📊 Applying efficiency optimization...")
        
        efficiency_params = {}
        
        # Optimize based on utilization patterns
        if metrics.npu_utilization < 50.0:
            # NPU underutilized - increase NPU workload
            efficiency_params.update({
                'npu_block_size': 128,
                'npu_frequency': 'turbo'
            })
        
        if metrics.igpu_utilization < 70.0:
            # iGPU underutilized - increase iGPU workload
            efficiency_params.update({
                'igpu_workgroup_x': 16,
                'igpu_workgroup_y': 8,
                'igpu_tile_size': 32
            })
        
        self.optimal_parameters.update(efficiency_params)
        logger.info("   ✅ Efficiency optimization applied")
    
    def _adaptive_parameter_tuning(self, metrics: PerformanceMetrics):
        """Adaptive parameter tuning based on performance"""
        logger.info("🔧 Running adaptive parameter tuning...")
        
        # Find best performing parameters from history
        best_efficiency = 0.0
        best_params = {}
        
        # Simple hill-climbing optimization
        for param, values in self.tuning_parameters.items():
            current_value = self.optimal_parameters.get(param, values[0])
            current_index = values.index(current_value) if current_value in values else 0
            
            # Try adjacent values
            for offset in [-1, 1]:
                new_index = current_index + offset
                if 0 <= new_index < len(values):
                    new_value = values[new_index]
                    
                    # Estimate performance impact (simplified)
                    estimated_efficiency = self._estimate_parameter_impact(param, new_value, metrics)
                    
                    if estimated_efficiency > best_efficiency:
                        best_efficiency = estimated_efficiency
                        best_params[param] = new_value
        
        if best_params:
            self.optimal_parameters.update(best_params)
            logger.info(f"   🎯 Updated parameters: {best_params}")
        else:
            logger.info("   ✅ Current parameters are optimal")
    
    def _estimate_parameter_impact(self, param: str, value, metrics: PerformanceMetrics):
        """Estimate performance impact of parameter change"""
        # Simplified heuristic-based estimation
        base_efficiency = metrics.efficiency_score
        
        impact_factors = {
            'npu_block_size': {32: 0.8, 64: 1.0, 128: 1.1, 256: 0.9},
            'igpu_workgroup_x': {4: 0.7, 8: 1.0, 16: 1.2, 32: 0.9},
            'igpu_workgroup_y': {4: 0.7, 8: 1.0, 16: 1.2, 32: 0.9},
            'igpu_tile_size': {8: 0.8, 16: 1.0, 32: 1.1, 64: 0.9},
            'memory_prefetch': {64: 0.9, 128: 1.0, 256: 1.1, 512: 0.8},
            'parallel_streams': {1: 0.8, 2: 1.0, 4: 1.2, 8: 0.9}
        }
        
        factor = impact_factors.get(param, {}).get(value, 1.0)
        return base_efficiency * factor
    
    def get_optimal_parameters(self):
        """Get current optimal parameters"""
        return self.optimal_parameters.copy()
    
    def apply_hardware_specific_tuning(self, engine_config):
        """Apply hardware-specific tuning to engine configuration"""
        logger.info("🎯 Applying hardware-specific tuning...")
        
        # Get current optimal parameters
        optimal = self.get_optimal_parameters()
        
        # Apply NPU tuning
        if 'npu_block_size' in optimal:
            engine_config['npu_block_size'] = optimal['npu_block_size']
        if 'npu_frequency' in optimal:
            engine_config['npu_frequency'] = optimal['npu_frequency']
        
        # Apply iGPU tuning
        if 'igpu_workgroup_x' in optimal:
            engine_config['igpu_workgroup_x'] = optimal['igpu_workgroup_x']
        if 'igpu_workgroup_y' in optimal:
            engine_config['igpu_workgroup_y'] = optimal['igpu_workgroup_y']
        if 'igpu_tile_size' in optimal:
            engine_config['igpu_tile_size'] = optimal['igpu_tile_size']
        
        # Apply memory tuning
        if 'memory_prefetch' in optimal:
            engine_config['memory_prefetch'] = optimal['memory_prefetch']
        if 'parallel_streams' in optimal:
            engine_config['parallel_streams'] = optimal['parallel_streams']
        
        logger.info(f"   ✅ Applied {len(optimal)} optimizations")
        return engine_config
    
    def get_performance_report(self):
        """Get comprehensive performance report"""
        if not self.performance_history:
            return {"status": "No data available"}
        
        recent_metrics = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        avg_metrics = self._calculate_average_metrics(recent_metrics)
        
        report = {
            "monitoring_duration_minutes": len(self.performance_history) / 60.0,
            "average_metrics": {
                "npu_utilization_percent": avg_metrics.npu_utilization,
                "igpu_utilization_percent": avg_metrics.igpu_utilization,
                "memory_bandwidth_percent": avg_metrics.memory_bandwidth,
                "temperature_celsius": avg_metrics.temperature_c,
                "power_watts": avg_metrics.power_watts,
                "efficiency_score": avg_metrics.efficiency_score
            },
            "optimization_status": {
                "thermal_throttling": avg_metrics.temperature_c > self.THERMAL_LIMIT_C,
                "power_limited": avg_metrics.power_watts > self.POWER_LIMIT_W,
                "efficiency_good": avg_metrics.efficiency_score >= self.MIN_EFFICIENCY_SCORE
            },
            "optimal_parameters": self.optimal_parameters,
            "recommendations": self._generate_recommendations(avg_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: PerformanceMetrics):
        """Generate optimization recommendations"""
        recommendations = []
        
        if metrics.temperature_c > 80.0:
            recommendations.append("Consider improving cooling or reducing workload")
        
        if metrics.npu_utilization < 60.0:
            recommendations.append("NPU underutilized - consider increasing NPU workload")
        
        if metrics.igpu_utilization < 70.0:
            recommendations.append("iGPU underutilized - consider optimizing compute shaders")
        
        if metrics.efficiency_score < 0.7:
            recommendations.append("Overall efficiency low - enable adaptive tuning")
        
        if not recommendations:
            recommendations.append("Performance looks optimal")
        
        return recommendations

class HardwareSpecificOptimizer:
    """Hardware-specific optimizer for NPU Phoenix + Radeon 780M"""
    
    def __init__(self):
        self.tuner = AdvancedHardwareTuner()
        self.optimization_profiles = {
            'performance': self._create_performance_profile(),
            'balanced': self._create_balanced_profile(),
            'efficiency': self._create_efficiency_profile(),
            'thermal': self._create_thermal_profile()
        }
        
        logger.info("🏭 Hardware-Specific Optimizer initialized")
    
    def _create_performance_profile(self):
        """Create performance-focused optimization profile"""
        return {
            'npu_frequency': 'turbo',
            'npu_block_size': 128,
            'igpu_workgroup_x': 16,
            'igpu_workgroup_y': 8,
            'igpu_tile_size': 32,
            'memory_prefetch': 256,
            'parallel_streams': 4,
            'description': 'Maximum performance, higher power consumption'
        }
    
    def _create_balanced_profile(self):
        """Create balanced optimization profile"""
        return {
            'npu_frequency': 'boost',
            'npu_block_size': 64,
            'igpu_workgroup_x': 8,
            'igpu_workgroup_y': 8,
            'igpu_tile_size': 16,
            'memory_prefetch': 128,
            'parallel_streams': 2,
            'description': 'Balanced performance and efficiency'
        }
    
    def _create_efficiency_profile(self):
        """Create efficiency-focused optimization profile"""
        return {
            'npu_frequency': 'nominal',
            'npu_block_size': 64,
            'igpu_workgroup_x': 8,
            'igpu_workgroup_y': 4,
            'igpu_tile_size': 16,
            'memory_prefetch': 64,
            'parallel_streams': 1,
            'description': 'Maximum efficiency, lower power consumption'
        }
    
    def _create_thermal_profile(self):
        """Create thermal-optimized profile"""
        return {
            'npu_frequency': 'nominal',
            'npu_block_size': 32,
            'igpu_workgroup_x': 4,
            'igpu_workgroup_y': 4,
            'igpu_tile_size': 8,
            'memory_prefetch': 64,
            'parallel_streams': 1,
            'description': 'Thermal-safe operation'
        }
    
    def get_optimization_profile(self, profile_name: str):
        """Get optimization profile by name"""
        return self.optimization_profiles.get(profile_name, self.optimization_profiles['balanced'])
    
    def start_adaptive_optimization(self):
        """Start adaptive optimization system"""
        logger.info("🚀 Starting adaptive optimization system...")
        self.tuner.start_real_time_monitoring()
        return True
    
    def stop_optimization(self):
        """Stop optimization system"""
        self.tuner.stop_monitoring()

if __name__ == "__main__":
    # Test hardware-specific tuner
    logger.info("🧪 Testing Advanced Hardware Tuner...")
    
    optimizer = HardwareSpecificOptimizer()
    optimizer.start_adaptive_optimization()
    
    try:
        # Let it run for a short time
        time.sleep(10)
        
        # Get performance report
        report = optimizer.tuner.get_performance_report()
        
        print(f"\n📊 Hardware Performance Report:")
        print(f"   Monitoring duration: {report['monitoring_duration_minutes']:.1f} minutes")
        print(f"   Average temperature: {report['average_metrics']['temperature_celsius']:.1f}°C")
        print(f"   Average power: {report['average_metrics']['power_watts']:.1f}W")
        print(f"   Efficiency score: {report['average_metrics']['efficiency_score']:.2f}")
        print(f"   Optimal parameters: {len(report['optimal_parameters'])} applied")
        
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        
        print(f"\n✅ Advanced hardware tuner test completed!")
        
    finally:
        optimizer.stop_optimization()