#!/usr/bin/env python3
"""
Gemma 3n E4B Terminal Chat - Direct Acceleration
Production-ready chat interface with real NPU+Vulkan acceleration
"""

import sys
import time
import logging
from typing import Dict, Any
from gemma3n_e4b_simple_acceleration import SimpleGemma3nE4BAcceleratedModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Gemma3nE4BTerminalChat:
    """Terminal chat interface with NPU+Vulkan acceleration"""
    
    def __init__(self):
        self.model = None
        self.conversation_history = []
        self.session_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "session_start": time.time()
        }
        
        # Colors for output
        self.COLORS = {
            'BLUE': '\033[94m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'RED': '\033[91m',
            'PURPLE': '\033[95m',
            'CYAN': '\033[96m',
            'WHITE': '\033[97m',
            'BOLD': '\033[1m',
            'END': '\033[0m'
        }
        
        self.initialize_model()
        
    def print_colored(self, text: str, color: str = 'WHITE'):
        """Print colored text"""
        print(f"{self.COLORS[color]}{text}{self.COLORS['END']}")
        
    def initialize_model(self):
        """Initialize the accelerated model"""
        self.print_colored("🦄 Initializing Gemma 3n E4B with NPU+Vulkan acceleration...", "PURPLE")
        self.print_colored("⏳ This may take a few moments...", "YELLOW")
        
        try:
            self.model = SimpleGemma3nE4BAcceleratedModel()
            self.print_colored("✅ Model initialized successfully!", "GREEN")
            
            # Show hardware status
            report = self.model.get_performance_report()
            npu_status = "✅ ENABLED" if report['acceleration']['npu_phoenix']['available'] else "❌ DISABLED"
            vulkan_status = "✅ ENABLED" if report['acceleration']['vulkan_radeon']['available'] else "❌ DISABLED"
            
            self.print_colored(f"🔥 NPU Phoenix: {npu_status}", "CYAN")
            self.print_colored(f"🔥 Vulkan iGPU: {vulkan_status}", "CYAN")
            print()
            
        except Exception as e:
            self.print_colored(f"❌ Error initializing model: {e}", "RED")
            sys.exit(1)
            
    def print_header(self):
        """Print chat header"""
        self.print_colored("🦄 Gemma 3n E4B Terminal Chat - Direct Acceleration", "PURPLE")
        self.print_colored("=" * 60, "PURPLE")
        self.print_colored("🚀 Real NPU+Vulkan hardware acceleration", "CYAN")
        self.print_colored("💡 Type 'exit' to quit, 'help' for commands", "CYAN")
        self.print_colored("=" * 60, "PURPLE")
        print()
        
    def format_prompt(self, user_input: str) -> str:
        """Format user input for the model"""
        return f"<|user|>\n{user_input}\n<|end|>\n<|assistant|>\n"
        
    def chat_loop(self):
        """Main chat loop"""
        self.print_header()
        
        while True:
            try:
                # Get user input
                user_input = input(f"{self.COLORS['BLUE']}👤 You: {self.COLORS['END']}").strip()
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    self.print_colored("👋 Goodbye!", "CYAN")
                    break
                elif user_input.lower() in ['help', 'h']:
                    self.show_help()
                    continue
                elif user_input.lower() in ['clear', 'c']:
                    self.clear_conversation()
                    continue
                elif user_input.lower() in ['stats', 's']:
                    self.show_stats()
                    continue
                elif user_input.lower() in ['status']:
                    self.show_performance_status()
                    continue
                elif not user_input:
                    continue
                    
                # Format prompt
                prompt = self.format_prompt(user_input)
                
                # Generate response
                print(f"{self.COLORS['GREEN']}🤖 Gemma: {self.COLORS['END']}", end="", flush=True)
                start_time = time.time()
                
                result = self.model.accelerated_generate(
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.7
                )
                
                response = result['generated_text'].strip()
                generation_time = time.time() - start_time
                
                # Display response
                print(response)
                
                # Show performance metrics
                tps = result['tokens_per_second']
                tokens = result['tokens_generated']
                
                self.print_colored(f"📊 {tokens} tokens | {tps:.1f} TPS | {generation_time:.1f}s", "YELLOW")
                
                # Update session stats
                self.session_stats["total_requests"] += 1
                self.session_stats["total_tokens"] += tokens
                self.session_stats["total_time"] += generation_time
                
                # Store in history
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': response,
                    'metrics': {
                        'tokens': tokens,
                        'tps': tps,
                        'time': generation_time
                    }
                })
                
                print()
                
            except KeyboardInterrupt:
                print()
                self.print_colored("👋 Goodbye!", "CYAN")
                break
            except Exception as e:
                print()
                self.print_colored(f"❌ Error: {e}", "RED")
                print()
                
    def show_help(self):
        """Show help information"""
        self.print_colored("\n🆘 Help Commands:", "CYAN")
        print("  help, h     - Show this help message")
        print("  stats, s    - Show session statistics")
        print("  status      - Show hardware performance status")
        print("  clear, c    - Clear conversation history")
        print("  exit, quit  - Exit chat")
        print()
        
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.print_colored("✅ Conversation cleared", "GREEN")
        
    def show_stats(self):
        """Show session statistics"""
        session_time = time.time() - self.session_stats["session_start"]
        avg_tps = self.session_stats["total_tokens"] / self.session_stats["total_time"] if self.session_stats["total_time"] > 0 else 0
        
        self.print_colored("\n📊 Session Statistics:", "CYAN")
        self.print_colored("=" * 30, "CYAN")
        print(f"  Total Requests: {self.session_stats['total_requests']}")
        print(f"  Total Tokens: {self.session_stats['total_tokens']}")
        print(f"  Total Response Time: {self.session_stats['total_time']:.1f}s")
        print(f"  Average TPS: {avg_tps:.1f}")
        print(f"  Session Duration: {session_time:.1f}s")
        print(f"  Conversation Length: {len(self.conversation_history)} exchanges")
        print()
        
    def show_performance_status(self):
        """Show detailed performance status"""
        if not self.model:
            self.print_colored("❌ Model not initialized", "RED")
            return
            
        report = self.model.get_performance_report()
        
        self.print_colored("\n🔥 Hardware Performance Status:", "CYAN")
        self.print_colored("=" * 40, "CYAN")
        
        # Hardware status
        npu_available = report['acceleration']['npu_phoenix']['available']
        vulkan_available = report['acceleration']['vulkan_radeon']['available']
        
        print(f"🔥 NPU Phoenix: {'✅ ENABLED' if npu_available else '❌ DISABLED'}")
        print(f"🔥 Vulkan iGPU: {'✅ ENABLED' if vulkan_available else '❌ DISABLED'}")
        
        # NPU details
        if npu_available:
            npu_status = report['acceleration']['npu_phoenix']
            print(f"   NPU Device: {npu_status.get('device', 'Unknown')}")
            print(f"   NPU Kernels: {npu_status.get('kernels_compiled', 0)} compiled")
            print(f"   NPU Utilization: {npu_status.get('utilization_percent', 0):.1f}%")
            
        # Vulkan details
        if vulkan_available:
            vulkan_status = report['acceleration']['vulkan_radeon']
            print(f"   Vulkan Device: {vulkan_status.get('device', 'Unknown')}")
            print(f"   Vulkan Shaders: {vulkan_status.get('shaders_compiled', 0)} compiled")
            print(f"   Vulkan Memory: {vulkan_status.get('memory_allocated_mb', 0):.1f}MB")
            
        # Performance metrics
        metrics = report['performance']
        print(f"⚡ Current TPS: {metrics['tokens_per_second']:.1f}")
        print(f"⏱️  Last inference: {metrics['total_inference_time']:.1f}s")
        
        # Model architecture
        print(f"🏗️  Model: {report['model']}")
        print(f"🧠 Architecture: Elastic 4B effective parameters")
        
        print()

def main():
    """Main function"""
    try:
        # Initialize chat
        chat = Gemma3nE4BTerminalChat()
        
        # Start chat loop
        chat.chat_loop()
        
        # Show final stats
        chat.show_stats()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()