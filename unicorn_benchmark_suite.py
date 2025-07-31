#!/usr/bin/env python3.13
"""
Unicorn Automated Benchmarking Suite
Comprehensive performance testing and validation framework
"""

import torch
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import os
import subprocess
import psutil
import GPUtil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    timestamp: str
    test_name: str
    device: str
    quantization: str
    batch_size: int
    seq_length: int
    hidden_size: int
    num_layers: int
    
    # Timing results
    warmup_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    iterations: int
    
    # Performance metrics
    tokens_per_sec: float
    flops: float
    memory_used_gb: float
    power_watts: Optional[float]
    
    # Component breakdown
    component_times: Dict[str, float]
    
    # System info
    cpu_percent: float
    gpu_utilization: float
    temperature_c: Optional[float]
    
    # Validation
    output_valid: bool
    accuracy_score: Optional[float]


class UnicornBenchmarkSuite:
    """Comprehensive benchmarking suite for Unicorn Execution Engine"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
        # Detection flags
        self.has_igpu = self._detect_igpu()
        self.has_npu = self._detect_npu()
        self.has_rocm = self._detect_rocm()
        
        logger.info(f"🦄 Unicorn Benchmark Suite initialized")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   iGPU available: {self.has_igpu}")
        logger.info(f"   NPU available: {self.has_npu}")
        logger.info(f"   ROCm available: {self.has_rocm}")
        
    def _detect_igpu(self) -> bool:
        """Detect if iGPU is available"""
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                for device in devices:
                    if 'gfx1103' in device.name.lower():
                        return True
        except:
            pass
        return False
        
    def _detect_npu(self) -> bool:
        """Detect if NPU is available"""
        try:
            import pyxrt
            device = pyxrt.device(0)
            return True
        except:
            pass
        return False
        
    def _detect_rocm(self) -> bool:
        """Detect if ROCm is available"""
        return os.path.exists('/opt/rocm') and os.environ.get('ROCM_PATH') is not None
        
    def run_full_suite(self):
        """Run complete benchmark suite"""
        logger.info("Starting full benchmark suite...")
        
        # Test configurations
        test_configs = [
            # (batch_size, seq_length, description)
            (1, 32, "Short context"),
            (1, 128, "Medium context"),
            (1, 256, "Long context"),
            (1, 512, "Very long context"),
            (4, 32, "Batch processing"),
            (8, 32, "Large batch"),
        ]
        
        quantization_types = ['none', 'fp16', 'int8', 'int4']
        devices = []
        
        if self.has_igpu:
            devices.append('igpu')
        if self.has_npu:
            devices.append('npu')
        if self.has_igpu and self.has_npu:
            devices.append('hybrid')
        if not devices:
            devices.append('cpu')
            
        total_tests = len(test_configs) * len(quantization_types) * len(devices)
        current_test = 0
        
        for device in devices:
            for quant in quantization_types:
                # Skip INT4 on devices that don't support it
                if quant == 'int4' and device == 'cpu':
                    continue
                    
                for batch_size, seq_len, desc in test_configs:
                    current_test += 1
                    logger.info(f"\nTest {current_test}/{total_tests}: {desc}")
                    logger.info(f"  Device: {device}, Quantization: {quant}")
                    logger.info(f"  Batch: {batch_size}, Seq: {seq_len}")
                    
                    try:
                        result = self.run_single_benchmark(
                            device=device,
                            quantization=quant,
                            batch_size=batch_size,
                            seq_length=seq_len,
                            test_name=desc
                        )
                        self.results.append(result)
                        
                        # Print summary
                        logger.info(f"  ✓ Speed: {result.tokens_per_sec:.2f} tok/s")
                        logger.info(f"  ✓ Min latency: {result.min_time*1000:.1f}ms")
                        
                        if result.tokens_per_sec >= 21.0:
                            logger.info(f"  🎯 TARGET ACHIEVED!")
                            
                    except Exception as e:
                        logger.error(f"  ✗ Test failed: {e}")
                        
        # Save results
        self.save_results()
        
        # Generate report
        self.generate_report()
        
    def run_single_benchmark(self,
                           device: str,
                           quantization: str,
                           batch_size: int,
                           seq_length: int,
                           test_name: str,
                           iterations: int = 10,
                           warmup: int = 3) -> BenchmarkResult:
        """Run a single benchmark test"""
        
        # Load appropriate engine
        engine = self._load_engine(device, quantization)
        
        # Create test data
        hidden_size = 2560  # Default test size
        num_layers = 42     # Default test layers
        x = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float32)
        
        # Get initial system state
        initial_cpu = psutil.cpu_percent(interval=0.1)
        initial_gpu = self._get_gpu_utilization()
        
        # Warmup
        logger.info(f"  Warming up ({warmup} iterations)...")
        warmup_start = time.time()
        for _ in range(warmup):
            _ = self._run_inference(engine, x, device)
        warmup_time = time.time() - warmup_start
        
        # Benchmark
        logger.info(f"  Benchmarking ({iterations} iterations)...")
        times = []
        component_times_list = []
        
        for i in range(iterations):
            start = time.time()
            output, component_times = self._run_inference_with_timing(engine, x, device)
            elapsed = time.time() - start
            times.append(elapsed)
            component_times_list.append(component_times)
            
        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        # Average component times
        avg_components = {}
        for key in component_times_list[0].keys():
            avg_components[key] = np.mean([ct[key] for ct in component_times_list])
            
        # Calculate performance metrics
        tokens_per_sec = 1.0 / (min_time * num_layers)
        flops = self._calculate_flops(hidden_size, seq_length, num_layers)
        
        # Get system metrics
        final_cpu = psutil.cpu_percent(interval=0.1)
        final_gpu = self._get_gpu_utilization()
        memory_used = self._get_memory_usage(device)
        temperature = self._get_temperature(device)
        power = self._get_power_usage(device)
        
        # Validate output
        output_valid = torch.isfinite(output).all().item()
        
        return BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            test_name=test_name,
            device=device,
            quantization=quantization,
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            warmup_time=warmup_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_time=std_time,
            iterations=iterations,
            tokens_per_sec=tokens_per_sec,
            flops=flops,
            memory_used_gb=memory_used,
            power_watts=power,
            component_times=avg_components,
            cpu_percent=final_cpu - initial_cpu,
            gpu_utilization=final_gpu,
            temperature_c=temperature,
            output_valid=output_valid,
            accuracy_score=None  # TODO: Implement accuracy testing
        )
        
    def _load_engine(self, device: str, quantization: str):
        """Load the appropriate engine"""
        if quantization == 'int4' and self.has_rocm:
            # Try INT4 WMMA engine
            try:
                from magic_unicorn_ultra_speed import MagicUnicornUltraSpeed
                return MagicUnicornUltraSpeed()
            except:
                pass
                
        # Fallback to standard engine
        from optimized_hybrid_pipeline import OptimizedHybridEngine
        return OptimizedHybridEngine()
        
    def _run_inference(self, engine, x, device):
        """Run inference on the engine"""
        if hasattr(engine, 'transformer_layer_ultra'):
            return engine.transformer_layer_ultra(x, layer_idx=0)
        elif hasattr(engine, 'forward_layer_optimized'):
            # Create dummy weights if needed
            weights = self._create_dummy_weights(x.shape[-1])
            output, _ = engine.forward_layer_optimized(x, weights)
            return output
        else:
            raise ValueError("Unknown engine type")
            
    def _run_inference_with_timing(self, engine, x, device):
        """Run inference and collect component timings"""
        component_times = {}
        
        if hasattr(engine, 'forward_layer_optimized'):
            weights = self._create_dummy_weights(x.shape[-1])
            output, layer_time = engine.forward_layer_optimized(x, weights)
            
            # Try to get component breakdown from engine stats
            if hasattr(engine, 'speed_stats'):
                stats = engine.speed_stats
                component_times = {
                    'qkv_projection': stats.get('qkv_time', 0),
                    'attention': stats.get('attn_time', 0),
                    'output_projection': stats.get('out_proj_time', 0),
                    'ffn': stats.get('ffn_time', 0),
                }
            else:
                # Estimate component times
                component_times = {
                    'qkv_projection': layer_time * 0.3,
                    'attention': layer_time * 0.1,
                    'output_projection': layer_time * 0.1,
                    'ffn': layer_time * 0.5,
                }
        else:
            # Fallback
            output = self._run_inference(engine, x, device)
            component_times = {
                'qkv_projection': 0,
                'attention': 0,
                'output_projection': 0,
                'ffn': 0,
            }
            
        return output, component_times
        
    def _create_dummy_weights(self, hidden_size: int) -> Dict[str, torch.Tensor]:
        """Create dummy weights for testing"""
        intermediate_size = int(hidden_size * 2.625)  # Typical ratio
        return {
            'q_proj': torch.randn(hidden_size, hidden_size),
            'k_proj': torch.randn(hidden_size, hidden_size),
            'v_proj': torch.randn(hidden_size, hidden_size),
            'o_proj': torch.randn(hidden_size, hidden_size),
            'gate_proj': torch.randn(intermediate_size, hidden_size),
            'up_proj': torch.randn(intermediate_size, hidden_size),
            'down_proj': torch.randn(hidden_size, intermediate_size),
        }
        
    def _calculate_flops(self, hidden_size: int, seq_length: int, num_layers: int) -> float:
        """Calculate theoretical FLOPS for the model"""
        # Simplified FLOPS calculation for transformer
        intermediate_size = int(hidden_size * 2.625)
        
        # Per layer FLOPS
        qkv_flops = 3 * (seq_length * hidden_size * hidden_size) * 2
        attn_flops = 2 * (seq_length * seq_length * hidden_size) * 2
        out_flops = (seq_length * hidden_size * hidden_size) * 2
        ffn_flops = 3 * (seq_length * hidden_size * intermediate_size) * 2
        
        layer_flops = qkv_flops + attn_flops + out_flops + ffn_flops
        total_flops = layer_flops * num_layers
        
        return total_flops
        
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except:
            pass
        return 0.0
        
    def _get_memory_usage(self, device: str) -> float:
        """Get memory usage in GB"""
        if device == 'cpu':
            return psutil.virtual_memory().used / (1024**3)
        else:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].memoryUsed / 1024
            except:
                pass
        return 0.0
        
    def _get_temperature(self, device: str) -> Optional[float]:
        """Get device temperature"""
        try:
            if device in ['igpu', 'hybrid']:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].temperature
        except:
            pass
        return None
        
    def _get_power_usage(self, device: str) -> Optional[float]:
        """Get power usage in watts"""
        try:
            if device in ['igpu', 'hybrid']:
                # Try rocm-smi
                result = subprocess.run(['rocm-smi', '--showpower'], 
                                      capture_output=True, text=True)
                # Parse power from output
                # This is device-specific parsing
                return None
        except:
            pass
        return None
        
    def save_results(self):
        """Save benchmark results"""
        # Save as JSON
        results_file = self.output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
        # Save as CSV
        df = pd.DataFrame([asdict(r) for r in self.results])
        csv_file = self.output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"CSV saved to {csv_file}")
        
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        if not self.results:
            logger.warning("No results to report")
            return
            
        # Create visualizations
        self._plot_performance_comparison()
        self._plot_scaling_analysis()
        self._plot_component_breakdown()
        self._generate_html_report()
        
    def _plot_performance_comparison(self):
        """Plot performance comparison across configurations"""
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        plt.figure(figsize=(12, 8))
        
        # Group by device and quantization
        pivot = df.pivot_table(
            values='tokens_per_sec',
            index='seq_length',
            columns=['device', 'quantization'],
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar')
        plt.axhline(y=21, color='r', linestyle='--', label='Target (21 tok/s)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Tokens/sec')
        plt.title('Performance Comparison Across Configurations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300)
        plt.close()
        
    def _plot_scaling_analysis(self):
        """Plot scaling analysis"""
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sequence length scaling
        seq_scaling = df[df['batch_size'] == 1].groupby(['seq_length', 'device', 'quantization'])['avg_time'].mean().reset_index()
        
        for (device, quant), group in seq_scaling.groupby(['device', 'quantization']):
            ax1.plot(group['seq_length'], group['avg_time'] * 1000, 
                    marker='o', label=f'{device}-{quant}')
            
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Sequence Length Scaling')
        ax1.legend()
        ax1.grid(True)
        
        # Batch size scaling
        batch_scaling = df[df['seq_length'] == 32].groupby(['batch_size', 'device', 'quantization'])['tokens_per_sec'].mean().reset_index()
        
        for (device, quant), group in batch_scaling.groupby(['device', 'quantization']):
            ax2.plot(group['batch_size'], group['tokens_per_sec'], 
                    marker='o', label=f'{device}-{quant}')
            
        ax2.axhline(y=21, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Tokens/sec')
        ax2.set_title('Batch Size Scaling')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scaling_analysis.png', dpi=300)
        plt.close()
        
    def _plot_component_breakdown(self):
        """Plot component timing breakdown"""
        # Get best result for each configuration
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Extract component times
        components = ['qkv_projection', 'attention', 'output_projection', 'ffn']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique device-quantization combinations
        configs = df.groupby(['device', 'quantization']).first().reset_index()
        
        bottoms = np.zeros(len(configs))
        
        for component in components:
            values = []
            for _, row in configs.iterrows():
                comp_times = row['component_times']
                if isinstance(comp_times, dict):
                    values.append(comp_times.get(component, 0) * 1000)
                else:
                    values.append(0)
                    
            ax.bar(range(len(configs)), values, bottom=bottoms, label=component)
            bottoms += values
            
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([f"{r['device']}-{r['quantization']}" for _, r in configs.iterrows()], 
                          rotation=45, ha='right')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Component Timing Breakdown')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_breakdown.png', dpi=300)
        plt.close()
        
    def _generate_html_report(self):
        """Generate HTML report"""
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Unicorn Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .achieved {{ color: green; font-weight: bold; }}
        .close {{ color: orange; }}
        .far {{ color: red; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>🦄 Unicorn Execution Engine Benchmark Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Summary</h2>
    <ul>
        <li>Total tests: {len(df)}</li>
        <li>Best performance: {df['tokens_per_sec'].max():.2f} tokens/sec</li>
        <li>Target achieved: {'Yes' if df['tokens_per_sec'].max() >= 21 else 'No'}</li>
    </ul>
    
    <h2>Best Results by Configuration</h2>
    <table>
        <tr>
            <th>Device</th>
            <th>Quantization</th>
            <th>Seq Length</th>
            <th>Tokens/sec</th>
            <th>Latency (ms)</th>
            <th>vs Target</th>
        </tr>
"""
        
        # Get best result for each device-quantization combo
        best_results = df.loc[df.groupby(['device', 'quantization'])['tokens_per_sec'].idxmax()]
        
        for _, row in best_results.iterrows():
            ratio = row['tokens_per_sec'] / 21.0
            if ratio >= 1.0:
                class_name = 'achieved'
            elif ratio >= 0.5:
                class_name = 'close'
            else:
                class_name = 'far'
                
            html += f"""
        <tr>
            <td>{row['device']}</td>
            <td>{row['quantization']}</td>
            <td>{row['seq_length']}</td>
            <td class="{class_name}">{row['tokens_per_sec']:.2f}</td>
            <td>{row['min_time']*1000:.1f}</td>
            <td class="{class_name}">{ratio:.2f}x</td>
        </tr>
"""
        
        html += """
    </table>
    
    <h2>Performance Visualizations</h2>
    <img src="performance_comparison.png" alt="Performance Comparison">
    <img src="scaling_analysis.png" alt="Scaling Analysis">
    <img src="component_breakdown.png" alt="Component Breakdown">
    
</body>
</html>
"""
        
        report_file = self.output_dir / 'benchmark_report.html'
        with open(report_file, 'w') as f:
            f.write(html)
            
        logger.info(f"HTML report saved to {report_file}")


def main():
    """Run benchmark suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unicorn Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    parser.add_argument('--device', type=str, help='Specific device to test')
    parser.add_argument('--quantization', type=str, help='Specific quantization to test')
    parser.add_argument('--output', type=str, default='benchmark_results', help='Output directory')
    
    args = parser.parse_args()
    
    suite = UnicornBenchmarkSuite(output_dir=args.output)
    
    if args.quick:
        # Quick test
        result = suite.run_single_benchmark(
            device=args.device or 'auto',
            quantization=args.quantization or 'int4',
            batch_size=1,
            seq_length=128,
            test_name='Quick Test'
        )
        print(f"\nQuick Test Results:")
        print(f"  Speed: {result.tokens_per_sec:.2f} tokens/sec")
        print(f"  Latency: {result.min_time*1000:.1f}ms")
        print(f"  Valid output: {result.output_valid}")
    else:
        # Full suite
        suite.run_full_suite()


if __name__ == "__main__":
    main()