#!/usr/bin/env python3
"""
Benchmark Optimized TPS with Weight Caching
"""

import numpy as np
import time
import logging
import requests
import json
import argparse

# Assuming the server is running on localhost:8006
SERVER_URL = "http://localhost:8006/v1/chat/completions"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_optimized_tps(batch_size: int, num_iterations: int = 20, max_tokens: int = 50):
    """Benchmark TPS by sending requests to the optimized API server."""

    logger.info("🚀 Benchmarking Optimized Pipeline via API Server")
    logger.info("=" * 60)
    logger.info(f"\n📊 Configuration:")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Iterations: {num_iterations}")
    logger.info(f"   Max tokens per generation: {max_tokens}")

    # Warmup
    logger.info(f"\n🔥 Warming up...")
    for _ in range(3):
        try:
            response = requests.post(
                SERVER_URL,
                headers={'Content-Type': 'application/json'},
                json={
                    "model": "gemma-3-27b-pure-hardware",
                    "messages": [{"role": "user", "content": "Hello"}] * batch_size, # Create a batch of identical messages
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Warmup request failed: {e}")
            return

    # Benchmark
    logger.info(f"\n⚡ Benchmarking...")
    total_generated_tokens = 0
    start_time = time.time()

    for i in range(num_iterations):
        try:
            response = requests.post(
                SERVER_URL,
                headers={'Content-Type': 'application/json'},
                json={
                    "model": "gemma-3-27b-pure-hardware",
                    "messages": [{"role": "user", "content": "Hello"}] * batch_size,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Assuming the server returns usage statistics
            if "usage" in result and "completion_tokens" in result["usage"]:
                total_generated_tokens += result["usage"]["completion_tokens"]
            else:
                # Fallback if usage stats are not directly available
                total_generated_tokens += batch_size * max_tokens # Estimate

        except requests.exceptions.RequestException as e:
            logger.error(f"Benchmark request failed at iteration {i}: {e}")
            break

    elapsed_time = time.time() - start_time

    # Calculate metrics
    tps = total_generated_tokens / elapsed_time if elapsed_time > 0 else 0

    logger.info(f"\n📊 Results:")
    logger.info(f"   Total generated tokens: {total_generated_tokens}")
    logger.info(f"   Total time: {elapsed_time:.2f}s")
    logger.info(f"   ⚡ Tokens per second (TPS): {tps:.2f}")

    logger.info(f"\n✅ Benchmarking complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark optimized TPS via API server.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--num-iterations", type=int, default=20, help="Number of benchmark iterations.")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate per request.")
    args = parser.parse_args()

    benchmark_optimized_tps(args.batch_size, args.num_iterations, args.max_tokens)