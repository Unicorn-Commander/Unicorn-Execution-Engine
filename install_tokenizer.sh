#!/bin/bash
# Install Gemma tokenizer requirements

echo "🔤 Installing tokenizer dependencies..."

# Install sentencepiece for Gemma tokenizer
pip3.13 install sentencepiece

# Install transformers with tokenizer support
pip3.13 install transformers

echo "✅ Tokenizer dependencies installed"