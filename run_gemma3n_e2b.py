#!/usr/bin/env python3
"""
Gemma 3n E2B NPU+iGPU Hybrid Execution
Main script for running Gemma 3n E2B with AMD NPU Phoenix + Radeon 780M hybrid execution
Target: 40-80 TPS, 20-40ms TTFT
"""

import os
import sys
import time
import argparse
import asyncio
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from gemma3n_e2b_loader import HybridConfig, Gemma3nE2BLoader
from hybrid_orchestrator import HybridOrchestrator, GenerationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print project banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 Gemma 3n E2B Hybrid NPU+iGPU Execution                  ║
║                                                                               ║
║  💻 Hardware: AMD NPU Phoenix (16 TOPS) + Radeon 780M iGPU                   ║
║  🎯 Target: 40-80 TPS, 20-40ms TTFT                                           ║
║  🧠 Model: Gemma 3n E2B (1.91B effective / 5B total parameters)              ║
║  ⚡ Architecture: MatFormer with elastic parameter scaling                     ║
║                                                                               ║
║  📊 NPU: Prefill & Attention Operations                                       ║
║  🎮 iGPU: Decode & Memory-Intensive Operations                                ║
║  🖥️  CPU: Orchestration & Sampling                                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Gemma 3n E2B with hybrid NPU+iGPU execution')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='google/gemma-2-2b',
                       help='Model ID or path (default: google/gemma-2-2b)')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Input prompt for text generation')
    
    # Generation settings
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum new tokens to generate (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling (default: 50)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p (nucleus) sampling (default: 0.9)')
    
    # Hardware configuration
    parser.add_argument('--npu-memory', type=int, default=2048,
                       help='NPU memory budget in MB (default: 2048)')
    parser.add_argument('--igpu-memory', type=int, default=8192,
                       help='iGPU memory budget in MB (default: 8192)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU execution (disable NPU/iGPU)')
    
    # Testing and benchmarking
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--dry-run', action='store_true',
                       help='Test setup without actual generation')
    parser.add_argument('--profile', action='store_true',
                       help='Enable detailed performance profiling')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Save output to file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()

def configure_logging(verbose: bool):
    """Configure logging based on verbosity"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Enable debug logging for our modules
        logging.getLogger('gemma3n_e2b_loader').setLevel(logging.DEBUG)
        logging.getLogger('hybrid_orchestrator').setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

def create_configs(args) -> tuple[HybridConfig, GenerationConfig]:
    """Create configuration objects from arguments"""
    # Hybrid configuration
    hybrid_config = HybridConfig(
        model_id=args.model,
        npu_memory_budget=args.npu_memory * 1024 * 1024,  # Convert MB to bytes
        igpu_memory_budget=args.igpu_memory * 1024 * 1024,
        igpu_device="cuda:0" if not args.force_cpu else "cpu"
    )
    
    # Generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True
    )
    
    return hybrid_config, generation_config

def print_system_info():
    """Print system and hardware information"""
    import psutil
    import platform
    
    print("\n📊 System Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  CPU: {platform.processor()}")
    print(f"  Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    memory = psutil.virtual_memory()
    print(f"  RAM: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")
    
    # Check for PyTorch and CUDA
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("  PyTorch: Not available")
    
    print()

def check_npu_status():
    """Check NPU availability and status"""
    print("🔍 Checking NPU Status:")
    
    try:
        import subprocess
        result = subprocess.run(['/opt/xilinx/xrt/bin/xrt-smi', 'examine'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and "NPU Phoenix" in result.stdout:
            print("  ✅ NPU Phoenix detected and available")
            
            # Check for turbo mode
            if "turbo" in result.stdout.lower():
                print("  ⚡ NPU turbo mode: ENABLED")
            else:
                print("  ⚠️  NPU turbo mode: Not detected")
                
            return True
        else:
            print("  ❌ NPU not detected or not available")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"  ❌ NPU check failed: {e}")
        return False

async def run_generation(orchestrator: HybridOrchestrator, prompt: str, 
                        generation_config: GenerationConfig, args) -> dict:
    """Run text generation with performance monitoring"""
    
    print(f"\n🎯 Generation Settings:")
    print(f"  Max tokens: {generation_config.max_new_tokens}")
    print(f"  Temperature: {generation_config.temperature}")
    print(f"  Top-k: {generation_config.top_k}")
    print(f"  Top-p: {generation_config.top_p}")
    
    print(f"\n📝 Prompt: {prompt}")
    print("\n🚀 Starting hybrid generation...\n")
    
    # Start generation
    start_time = time.time()
    
    try:
        result = await orchestrator.generate_text(prompt, generation_config)
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Generation completed in {total_time:.2f}s")
        print(f"\n📄 Generated Text:")
        print("─" * 80)
        print(result['generated_text'])
        print("─" * 80)
        
        # Performance metrics
        metrics = result['metrics']
        print(f"\n📊 Performance Metrics:")
        print(f"  Time to First Token (TTFT): {metrics.ttft_ms:.1f}ms")
        print(f"  Tokens per Second (TPS): {metrics.tps:.1f}")
        print(f"  Total tokens generated: {len(result['generated_tokens'])}")
        print(f"  Average decode time: {metrics.decode_time_ms:.2f}ms/token")
        print(f"  Memory usage: {metrics.memory_usage_mb:.1f}MB")
        
        # Target analysis
        config = orchestrator.config
        ttft_ok = config.target_ttft_ms[0] <= metrics.ttft_ms <= config.target_ttft_ms[1]
        tps_ok = config.target_tps[0] <= metrics.tps <= config.target_tps[1]
        
        print(f"\n🎯 Target Analysis:")
        print(f"  TTFT target ({config.target_ttft_ms[0]}-{config.target_ttft_ms[1]}ms): {'✅' if ttft_ok else '❌'} ({metrics.ttft_ms:.1f}ms)")
        print(f"  TPS target ({config.target_tps[0]}-{config.target_tps[1]}): {'✅' if tps_ok else '❌'} ({metrics.tps:.1f})")
        
        if ttft_ok and tps_ok:
            print("  🎉 All performance targets met!")
        else:
            print("  ⚠️  Some targets not met - consider optimization")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise

async def run_benchmark(orchestrator: HybridOrchestrator, args):
    """Run comprehensive performance benchmark"""
    print("\n🏃‍♂️ Running Performance Benchmark...")
    
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "Climate change represents one of the most significant challenges",
        "The development of quantum computing will revolutionize"
    ]
    
    benchmark_results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Benchmark {i}/{len(test_prompts)}: '{prompt[:30]}...'")
        
        config = GenerationConfig(
            max_new_tokens=50,  # Shorter for benchmark
            temperature=0.7,
            do_sample=True
        )
        
        try:
            result = await orchestrator.generate_text(prompt, config)
            benchmark_results.append(result['metrics'])
            
            print(f"  TTFT: {result['metrics'].ttft_ms:.1f}ms, TPS: {result['metrics'].tps:.1f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    if benchmark_results:
        # Calculate averages
        avg_ttft = sum(m.ttft_ms for m in benchmark_results) / len(benchmark_results)
        avg_tps = sum(m.tps for m in benchmark_results) / len(benchmark_results)
        avg_memory = sum(m.memory_usage_mb for m in benchmark_results) / len(benchmark_results)
        
        print(f"\n📊 Benchmark Summary:")
        print(f"  Tests completed: {len(benchmark_results)}/{len(test_prompts)}")
        print(f"  Average TTFT: {avg_ttft:.1f}ms")
        print(f"  Average TPS: {avg_tps:.1f}")
        print(f"  Average memory: {avg_memory:.1f}MB")
        
        # Performance score
        config = orchestrator.config
        ttft_score = 100 if config.target_ttft_ms[0] <= avg_ttft <= config.target_ttft_ms[1] else 0
        tps_score = 100 if config.target_tps[0] <= avg_tps <= config.target_tps[1] else 0
        overall_score = (ttft_score + tps_score) / 2
        
        print(f"  Performance score: {overall_score:.0f}/100")
    else:
        print("  ❌ No successful benchmark runs")

def save_output(result: dict, output_path: str):
    """Save generation result to file"""
    try:
        output_data = {
            'generated_text': result['generated_text'],
            'metrics': {
                'ttft_ms': result['metrics'].ttft_ms,
                'tps': result['metrics'].tps,
                'memory_usage_mb': result['metrics'].memory_usage_mb,
                'tokens_generated': len(result['generated_tokens'])
            },
            'timestamp': time.time()
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Output saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Failed to save output: {e}")

async def main():
    """Main execution function"""
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    configure_logging(args.verbose)
    
    # System information
    print_system_info()
    
    # Check NPU status
    npu_available = check_npu_status()
    
    if not npu_available and not args.force_cpu:
        print("⚠️  NPU not available, falling back to CPU execution")
        args.force_cpu = True
    
    # Create configurations
    hybrid_config, generation_config = create_configs(args)
    
    if args.dry_run:
        print("\n🧪 Dry run mode - testing setup only")
        print(f"  Model: {hybrid_config.model_id}")
        print(f"  NPU memory budget: {hybrid_config.npu_memory_budget / (1024**2):.0f}MB")
        print(f"  iGPU memory budget: {hybrid_config.igpu_memory_budget / (1024**2):.0f}MB")
        print(f"  Target TTFT: {hybrid_config.target_ttft_ms[0]}-{hybrid_config.target_ttft_ms[1]}ms")
        print(f"  Target TPS: {hybrid_config.target_tps[0]}-{hybrid_config.target_tps[1]}")
        print("\n✅ Configuration validated successfully")
        return
    
    try:
        # Initialize loader
        print("\n🔄 Initializing Gemma 3n E2B loader...")
        loader = Gemma3nE2BLoader(hybrid_config)
        
        # Load model (this might take time)
        print("📦 Loading model and tokenizer...")
        model, tokenizer = loader.load_model()
        
        # Create partitions
        print("🔀 Partitioning model for hybrid execution...")
        partitions = loader.partition_for_hybrid_execution()
        
        # Performance estimation
        performance = loader.estimate_performance()
        print(f"\n📈 Performance Estimation:")
        print(f"  Estimated TTFT: {performance['estimated_ttft_ms']:.1f}ms")
        print(f"  Estimated TPS: {performance['estimated_tps']:.1f}")
        print(f"  NPU memory usage: {performance['memory_usage_npu_gb']:.2f}GB")
        print(f"  iGPU memory usage: {performance['memory_usage_igpu_gb']:.2f}GB")
        
        # Initialize orchestrator
        print("🎭 Initializing hybrid orchestrator...")
        orchestrator = HybridOrchestrator(hybrid_config, partitions)
        
        # Run benchmark if requested
        if args.benchmark:
            await run_benchmark(orchestrator, args)
            if not args.prompt:
                return
        
        # Generate text
        if args.prompt:
            result = await run_generation(orchestrator, args.prompt, generation_config, args)
            
            # Save output if requested
            if args.output:
                save_output(result, args.output)
        
        print("\n🎉 Execution completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Run main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)