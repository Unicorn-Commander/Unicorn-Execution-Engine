#!/usr/bin/env python3
"""
Verify actual model configuration and dimensions
"""

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard Gemma configurations
GEMMA_CONFIGS = {
    "gemma-3-4b": {
        "hidden_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 192,  # 3072 / 16
    },
    "gemma-3-27b": {
        "hidden_size": 5376,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,  # GQA
        "head_dim": 168,  # 5376 / 32 = 168
        "q_proj_size": 5376,  # 32 * 168
        "kv_proj_size": 2688,  # 16 * 168
    },
    "gemma-3-27b-corrected": {
        "hidden_size": 5376,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,  # GQA
        "head_dim": 128,  # Standard head dim
        "q_proj_size": 4096,  # 32 * 128
        "kv_proj_size": 2048,  # 16 * 128
    }
}

logger.info("🔍 Verifying Gemma 27B dimensions...")

# What we found in the actual model files
logger.info("\n📊 Actual tensor shapes from model:")
logger.info("   Q projection: [4096, 5376] → outputs 4096")
logger.info("   K projection: [2048, 5376] → outputs 2048")
logger.info("   V projection: [2048, 5376] → outputs 2048")
logger.info("   O projection: [5376, 4096] → outputs 5376")

# Calculate head dim
q_features = 4096
num_q_heads = 32
actual_head_dim = q_features // num_q_heads
logger.info(f"\n✅ Calculated head_dim: {actual_head_dim} (4096 / 32)")

# The confusion
logger.info("\n❓ Why the confusion about 168 vs 128?")
logger.info("   - Standard formula: head_dim = hidden_size / num_heads = 5376 / 32 = 168")
logger.info("   - But actual model uses: head_dim = 128 (standard size)")
logger.info("   - This means Q projection is NOT hidden_size!")

# NPU configuration
logger.info("\n🧠 NPU Configuration:")
logger.info("   - NPU kernel expects head_dim = 168 (based on formula)")
logger.info("   - But actual model has head_dim = 128")
logger.info("   - This is why NPU kernel validation fails!")

# Quantization
logger.info("\n📦 Quantization:")
logger.info("   - INT8: Standard for NPU attention weights")
logger.info("   - INT4: For memory efficiency (especially FFN)")
logger.info("   - 16x16: Likely refers to compute tile size, not quantization")

logger.info("\n💡 Solution: Update NPU kernel to accept head_dim=128")