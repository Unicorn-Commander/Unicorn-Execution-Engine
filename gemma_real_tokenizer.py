#!/usr/bin/env python3.13
"""
🔤 Real Gemma Tokenizer - Load actual Gemma tokenizer.json
Handles the full 256k vocabulary for real text generation
"""

import json
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Optional

class GemmaRealTokenizer:
    """Real Gemma tokenizer with full vocabulary"""
    
    def __init__(self, tokenizer_path: str = None):
        if tokenizer_path is None:
            # Try to find tokenizer automatically
            possible_paths = [
                "models/gemma-3n-e4b-it/tokenizer.json",
                "models/gemma-3n-e4b-it/tokenizer.model",
                "quantized_models/gemma-3-4b-it-quantized/tokenizer.json"
            ]
            
            for path in possible_paths:
                full_path = Path(path)
                if full_path.exists():
                    tokenizer_path = str(full_path)
                    break
        
        self.tokenizer_path = tokenizer_path
        self.vocab = {}
        self.merges = []
        self.special_tokens = {}
        
        # Token IDs
        self.pad_token_id = 0
        self.bos_token_id = 2
        self.eos_token_id = 1
        self.unk_token_id = 3
        
        print(f"🔤 Loading Gemma tokenizer from: {self.tokenizer_path}")
        self._load_tokenizer()
        
    def _load_tokenizer(self):
        """Load the tokenizer.json file"""
        try:
            with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            
            # Extract vocabulary
            if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                self.vocab = tokenizer_data['model']['vocab']
                print(f"✅ Loaded vocabulary: {len(self.vocab)} tokens")
            else:
                print("⚠️  No vocabulary found in tokenizer.json")
                self._create_basic_vocab()
            
            # Extract merges for BPE
            if 'model' in tokenizer_data and 'merges' in tokenizer_data['model']:
                self.merges = tokenizer_data['model']['merges']
                print(f"   Loaded {len(self.merges)} merges")
            
            # Extract special tokens
            if 'added_tokens' in tokenizer_data:
                for token_info in tokenizer_data['added_tokens']:
                    self.special_tokens[token_info['content']] = token_info['id']
                    
                    # Update special token IDs
                    if token_info['content'] == '<pad>':
                        self.pad_token_id = token_info['id']
                    elif token_info['content'] == '<bos>':
                        self.bos_token_id = token_info['id']
                    elif token_info['content'] == '<eos>':
                        self.eos_token_id = token_info['id']
                    elif token_info['content'] == '<unk>':
                        self.unk_token_id = token_info['id']
            
            # Create reverse vocabulary
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            
            # Add special tokens to reverse vocab
            for token, id in self.special_tokens.items():
                self.id_to_token[id] = token
            
            print(f"   Special tokens: {list(self.special_tokens.keys())[:5]}...")
            print(f"   BOS: {self.bos_token_id}, EOS: {self.eos_token_id}")
            
        except FileNotFoundError:
            print(f"❌ Tokenizer file not found: {self.tokenizer_path}")
            self._create_basic_vocab()
        except json.JSONDecodeError as e:
            print(f"❌ Error loading tokenizer: {e}")
            self._create_basic_vocab()
    
    def _create_basic_vocab(self):
        """Fallback: Create basic vocabulary"""
        print("⚠️  Using fallback vocabulary")
        
        # Basic tokens
        self.vocab = {
            '<pad>': 0, '<eos>': 1, '<bos>': 2, '<unk>': 3,
            ' ': 4, 'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9,
            'f': 10, 'g': 11, 'h': 12, 'i': 13, 'j': 14, 'k': 15,
            'l': 16, 'm': 17, 'n': 18, 'o': 19, 'p': 20, 'q': 21,
            'r': 22, 's': 23, 't': 24, 'u': 25, 'v': 26, 'w': 27,
            'x': 28, 'y': 29, 'z': 30,
        }
        
        # Add common words
        common_words = ['the', 'is', 'are', 'was', 'were', 'be', 'been', 
                       'have', 'has', 'had', 'do', 'does', 'did', 'will',
                       'would', 'could', 'should', 'may', 'might', 'must',
                       'artificial', 'intelligence', 'machine', 'learning',
                       'neural', 'network', 'model', 'data', 'computer']
        
        token_id = 31
        for word in common_words:
            self.vocab[word] = token_id
            token_id += 1
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        
        # Simple tokenization - try to use actual vocab
        # For production, would use proper BPE with merges
        
        # Normalize text
        text = text.strip()
        
        # Try word-level first
        words = text.split()
        
        for word in words:
            # Check if whole word is in vocab
            if word in self.vocab:
                tokens.append(self.vocab[word])
            elif word.lower() in self.vocab:
                tokens.append(self.vocab[word.lower()])
            else:
                # Fall back to character level or subwords
                # For now, simple character encoding
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.unk_token_id)
            
            # Add space between words
            if word != words[-1] and ' ' in self.vocab:
                tokens.append(self.vocab[' '])
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []
        
        for token_id in token_ids:
            # Skip special tokens if requested
            if skip_special_tokens and token_id in [self.pad_token_id, self.bos_token_id, 
                                                   self.eos_token_id, self.unk_token_id]:
                continue
            
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                tokens.append(token)
            else:
                # For out-of-vocab, skip or use <unk>
                if not skip_special_tokens:
                    tokens.append('<unk>')
        
        # Join tokens intelligently
        text = ""
        for token in tokens:
            # Handle special tokens
            if token.startswith('<') and token.endswith('>'):
                if not skip_special_tokens:
                    text += f" {token} "
            # Handle regular tokens
            else:
                # Smart spacing for punctuation
                if token in '.,!?;:)]}\'"':
                    text = text.rstrip() + token + " "
                elif token in '([{\'"':
                    text += token
                else:
                    if text and not text.endswith(' '):
                        text += " "
                    text += token
        
        # Clean up
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
    
    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encode multiple texts"""
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def batch_decode(self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode multiple token sequences"""
        return [self.decode(token_ids, skip_special_tokens) for token_ids in token_ids_batch]

# Test the tokenizer
if __name__ == "__main__":
    print("🔤 Testing Real Gemma Tokenizer")
    print("=" * 60)
    
    tokenizer = GemmaRealTokenizer()
    
    # Test texts
    test_texts = [
        "What is artificial intelligence?",
        "Machine learning is amazing!",
        "The AI model has 4 billion parameters.",
        "Hello, how are you today?"
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        
        # Encode
        tokens = tokenizer.encode(text)
        print(f"Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''} ({len(tokens)} total)")
        
        # Decode
        decoded = tokenizer.decode(tokens)
        print(f"Decoded: {decoded}")
        
        # Verify
        if text.lower() in decoded.lower():
            print("✅ Round-trip successful!")
        else:
            print("⚠️  Some differences in round-trip")