#!/usr/bin/env python3
"""
Gemma 3n E4B Terminal Chat Interface
Interactive chat with the Gemma 3n E4B API server with performance metrics
"""

import sys
import time
import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime
import threading

class Gemma3nE4BTerminalChat:
    """Terminal chat interface for Gemma 3n E4B API server"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.model = "gemma-3n-e4b-it"
        self.conversation_history = []
        self.session_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "session_start": time.time()
        }
        
        # Chat settings
        self.max_tokens = 150
        self.temperature = 0.7
        self.streaming = True
        self.debug = False
        
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
            'UNDERLINE': '\033[4m',
            'END': '\033[0m'
        }
    
    def print_colored(self, text: str, color: str = 'WHITE'):
        """Print colored text"""
        print(f"{self.COLORS[color]}{text}{self.COLORS['END']}")
    
    def print_header(self):
        """Print chat header"""
        print(f"{self.COLORS['PURPLE']}")
        print("🦄 Gemma 3n E4B Terminal Chat")
        print("=" * 50)
        print(f"🌐 Server: {self.base_url}")
        print(f"🤖 Model: {self.model}")
        print(f"⚙️  Max Tokens: {self.max_tokens}")
        print(f"🌡️  Temperature: {self.temperature}")
        print(f"📡 Streaming: {'ON' if self.streaming else 'OFF'}")
        print("=" * 50)
        print(f"{self.COLORS['END']}")
        print(f"{self.COLORS['CYAN']}Commands:{self.COLORS['END']}")
        print("  /help    - Show help")
        print("  /stats   - Show session statistics")
        print("  /health  - Check server health")
        print("  /settings - Change chat settings")
        print("  /clear   - Clear conversation")
        print("  /quit    - Exit chat")
        print()
    
    def check_server_health(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get("model_ready", False)
            return False
        except Exception:
            return False
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get detailed server status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}
    
    def send_chat_message(self, message: str) -> Dict[str, Any]:
        """Send a chat message to the API"""
        # Prepare conversation context
        messages = self.conversation_history + [{"role": "user", "content": message}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.streaming
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60,
                stream=self.streaming
            )
            
            if response.status_code != 200:
                return {
                    "error": f"API Error {response.status_code}: {response.text}",
                    "response_time": time.time() - start_time
                }
            
            if self.streaming:
                result = self.handle_streaming_response(response, start_time)
                # If streaming fails, fall back to regular mode
                if "error" in result and "streaming" in result["error"].lower():
                    self.print_colored("⚠️  Streaming failed, falling back to regular mode", "YELLOW")
                    self.streaming = False
                    # Retry with regular mode
                    return self.send_chat_message(message.split(":")[-1] if ":" in message else message)
                return result
            else:
                return self.handle_regular_response(response, start_time)
                
        except Exception as e:
            return {
                "error": str(e),
                "response_time": time.time() - start_time
            }
    
    def handle_regular_response(self, response: requests.Response, start_time: float) -> Dict[str, Any]:
        """Handle regular (non-streaming) response"""
        try:
            data = response.json()
            response_time = time.time() - start_time
            
            if "choices" in data and len(data["choices"]) > 0:
                assistant_message = data["choices"][0]["message"]["content"]
                
                # Calculate tokens per second
                tokens_generated = data.get("usage", {}).get("completion_tokens", 0)
                tps = tokens_generated / response_time if response_time > 0 else 0
                
                return {
                    "message": assistant_message,
                    "tokens_generated": tokens_generated,
                    "response_time": response_time,
                    "tokens_per_second": tps,
                    "usage": data.get("usage", {})
                }
            else:
                return {"error": "No response from model"}
                
        except Exception as e:
            return {"error": f"Failed to parse response: {e}"}
    
    def handle_streaming_response(self, response: requests.Response, start_time: float) -> Dict[str, Any]:
        """Handle streaming response"""
        try:
            assistant_message = ""
            tokens_generated = 0
            
            print(f"{self.COLORS['GREEN']}Assistant: {self.COLORS['END']}", end='', flush=True)
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            
                            # More robust null checks
                            if (chunk_data and 
                                isinstance(chunk_data, dict) and 
                                "choices" in chunk_data and 
                                chunk_data["choices"] and 
                                len(chunk_data["choices"]) > 0):
                                
                                choice = chunk_data["choices"][0]
                                
                                # Check if choice is not None and has delta
                                if (choice and 
                                    isinstance(choice, dict) and 
                                    "delta" in choice and 
                                    choice["delta"]):
                                    
                                    delta = choice["delta"]
                                    
                                    # Check if delta is not None and has content
                                    if (isinstance(delta, dict) and 
                                        "content" in delta and 
                                        delta["content"] is not None):
                                        
                                        content = delta["content"]
                                        if content:  # Only add non-empty content
                                            assistant_message += content
                                            print(content, end='', flush=True)
                                            
                                            # Rough token estimation (words * 1.3)
                                            tokens_generated = len(assistant_message.split()) * 1.3
                                
                                # Check for final chunk with usage
                                if (chunk_data and 
                                    isinstance(chunk_data, dict) and 
                                    "usage" in chunk_data and 
                                    chunk_data["usage"] and 
                                    isinstance(chunk_data["usage"], dict)):
                                    
                                    usage = chunk_data["usage"]
                                    if "completion_tokens" in usage and usage["completion_tokens"] is not None:
                                        tokens_generated = usage["completion_tokens"]
                                
                        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as parse_error:
                            # Debug logging for parsing errors
                            if self.debug:
                                print(f"\n[DEBUG] Parse error: {parse_error}")
                                print(f"[DEBUG] Data: {data_str[:100]}...")
                            continue
            
            print()  # New line after streaming
            
            response_time = time.time() - start_time
            tps = tokens_generated / response_time if response_time > 0 else 0
            
            return {
                "message": assistant_message,
                "tokens_generated": int(tokens_generated),
                "response_time": response_time,
                "tokens_per_second": tps,
                "streaming": True
            }
            
        except Exception as e:
            return {"error": f"Streaming error: {e}"}
    
    def show_stats(self):
        """Show session statistics"""
        session_time = time.time() - self.session_stats["session_start"]
        avg_tps = self.session_stats["total_tokens"] / self.session_stats["total_time"] if self.session_stats["total_time"] > 0 else 0
        
        self.print_colored("\n📊 Session Statistics:", "CYAN")
        print(f"  Total Requests: {self.session_stats['total_requests']}")
        print(f"  Total Tokens: {self.session_stats['total_tokens']}")
        print(f"  Total Response Time: {self.session_stats['total_time']:.2f}s")
        print(f"  Average TPS: {avg_tps:.1f}")
        print(f"  Session Duration: {session_time:.1f}s")
        print(f"  Conversation Length: {len(self.conversation_history)} messages")
        print()
    
    def show_health(self):
        """Show server health information"""
        self.print_colored("\n🏥 Server Health:", "CYAN")
        
        status = self.get_server_status()
        
        if not status:
            self.print_colored("❌ Server not responding", "RED")
            return
        
        # Basic status
        model_ready = status.get("model_ready", False)
        status_text = "✅ READY" if model_ready else "❌ NOT READY"
        print(f"  Status: {status_text}")
        print(f"  Model: {status.get('model_id', 'Unknown')}")
        print(f"  Loader State: {status.get('loader_state', 'Unknown')}")
        
        # Metrics
        metrics = status.get("metrics", {})
        print(f"  Requests: {metrics.get('requests_successful', 0)}/{metrics.get('requests_total', 0)}")
        print(f"  Average TPS: {metrics.get('average_tokens_per_second', 0):.1f}")
        print(f"  Active Elastic Params: {metrics.get('active_elastic_params', 0)}")
        print(f"  Memory Usage: {metrics.get('memory_usage', 0) / 1024**3:.1f}GB")
        
        # Components
        components = status.get("components", {})
        print(f"  Components: {sum(components.values())}/{len(components)} active")
        
        print()
    
    def show_settings(self):
        """Show and allow changing settings"""
        self.print_colored("\n⚙️  Current Settings:", "CYAN")
        print(f"  1. Max Tokens: {self.max_tokens}")
        print(f"  2. Temperature: {self.temperature}")
        print(f"  3. Streaming: {'ON' if self.streaming else 'OFF'}")
        print()
        
        try:
            choice = input("Enter setting number to change (or press Enter to skip): ").strip()
            
            if choice == "1":
                new_tokens = int(input(f"Enter new max tokens (current: {self.max_tokens}): "))
                self.max_tokens = max(1, min(4096, new_tokens))
                self.print_colored(f"✅ Max tokens set to {self.max_tokens}", "GREEN")
            
            elif choice == "2":
                new_temp = float(input(f"Enter new temperature (current: {self.temperature}): "))
                self.temperature = max(0.0, min(2.0, new_temp))
                self.print_colored(f"✅ Temperature set to {self.temperature}", "GREEN")
            
            elif choice == "3":
                self.streaming = not self.streaming
                self.print_colored(f"✅ Streaming {'enabled' if self.streaming else 'disabled'}", "GREEN")
        
        except (ValueError, KeyboardInterrupt):
            self.print_colored("❌ Invalid input", "RED")
    
    def show_help(self):
        """Show help information"""
        self.print_colored("\n🆘 Help:", "CYAN")
        print("  Commands:")
        print("    /help    - Show this help message")
        print("    /stats   - Show session statistics")
        print("    /health  - Check server health")
        print("    /settings - Change chat settings")
        print("    /clear   - Clear conversation history")
        print("    /quit    - Exit chat")
        print()
        print("  Chat Features:")
        print("    - Maintains conversation context")
        print("    - Real-time streaming responses")
        print("    - Performance metrics (TPS)")
        print("    - Colored output")
        print()
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.print_colored("✅ Conversation cleared", "GREEN")
    
    def run(self):
        """Main chat loop"""
        self.print_header()
        
        # Check server health
        if not self.check_server_health():
            self.print_colored("❌ Server is not ready. Please start the server first.", "RED")
            self.print_colored("   Try: ./start_gemma3n_e4b.sh api", "YELLOW")
            return
        
        self.print_colored("✅ Server is ready! Type your message or use /help for commands.\n", "GREEN")
        
        try:
            while True:
                # Get user input
                user_input = input(f"{self.COLORS['BLUE']}You: {self.COLORS['END']}").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command == 'quit' or command == 'q':
                        break
                    elif command == 'help' or command == 'h':
                        self.show_help()
                    elif command == 'stats':
                        self.show_stats()
                    elif command == 'health':
                        self.show_health()
                    elif command == 'settings':
                        self.show_settings()
                    elif command == 'clear':
                        self.clear_conversation()
                    elif command == 'test':
                        self.run_test()
                    elif command == 'debug':
                        self.debug = not self.debug
                        self.print_colored(f"🐛 Debug mode {'ON' if self.debug else 'OFF'}", "CYAN")
                    else:
                        self.print_colored(f"❌ Unknown command: {command}", "RED")
                    
                    continue
                
                # Send message to API
                result = self.send_chat_message(user_input)
                
                if "error" in result:
                    self.print_colored(f"❌ Error: {result['error']}", "RED")
                    continue
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": result["message"]})
                
                # Update session stats
                self.session_stats["total_requests"] += 1
                self.session_stats["total_tokens"] += result.get("tokens_generated", 0)
                self.session_stats["total_time"] += result.get("response_time", 0)
                
                # Show performance metrics
                if not self.streaming:
                    print(f"{self.COLORS['GREEN']}Assistant: {self.COLORS['END']}{result['message']}")
                
                # Performance info
                tps = result.get("tokens_per_second", 0)
                tokens = result.get("tokens_generated", 0)
                response_time = result.get("response_time", 0)
                
                self.print_colored(f"📊 {tps:.1f} TPS | {tokens} tokens | {response_time:.2f}s", "YELLOW")
                print()
        
        except KeyboardInterrupt:
            pass
        
        # Show final stats
        self.print_colored("\n👋 Chat session ended", "CYAN")
        self.show_stats()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemma 3n E4B Terminal Chat")
    parser.add_argument("--url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--max-tokens", type=int, default=150, help="Maximum tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming")
    
    args = parser.parse_args()
    
    chat = Gemma3nE4BTerminalChat(args.url)
    chat.max_tokens = args.max_tokens
    chat.temperature = args.temperature
    chat.streaming = not args.no_streaming
    
    chat.run()

if __name__ == "__main__":
    main()