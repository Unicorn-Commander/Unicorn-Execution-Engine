#!/usr/bin/env python3.13
"""
🦄 Unicorn Execution Engine GUI - Model Management Interface
Complete GUI for model management, chat, and performance monitoring
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import json
import threading
import time
from datetime import datetime
import psutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class UnicornGUI:
    """
    🦄 Unicorn Execution Engine GUI
    Complete interface for inference engine management
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🦄 Unicorn Execution Engine - Model Management")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')
        
        # API settings
        self.api_base = "http://localhost:8000"
        
        # Performance tracking
        self.performance_history = {"4b": [], "27b": []}
        self.system_metrics = {"cpu": [], "memory": [], "time": []}
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.start_monitoring()
        
        print("🦄 GUI initialized successfully")
    
    def setup_styles(self):
        """Setup custom styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark theme colors
        style.configure('Dark.TFrame', background='#2d2d2d')
        style.configure('Dark.TLabel', background='#2d2d2d', foreground='#ffffff')
        style.configure('Dark.TButton', background='#404040', foreground='#ffffff')
        style.configure('Header.TLabel', background='#2d2d2d', foreground='#00ff88', font=('Arial', 14, 'bold'))
        style.configure('Status.TLabel', background='#2d2d2d', foreground='#ffaa00', font=('Arial', 10))
        
    def create_widgets(self):
        """Create main GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, 
                              text="🦄 Unicorn Execution Engine", 
                              style='Header.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_chat_tab()
        self.create_models_tab()
        self.create_performance_tab()
        self.create_system_tab()
    
    def create_chat_tab(self):
        """Create chat interface tab"""
        chat_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(chat_frame, text="💬 Chat")
        
        # Chat history
        history_label = ttk.Label(chat_frame, text="Chat History:", style='Dark.TLabel')
        history_label.pack(anchor=tk.W, pady=(10, 5))
        
        self.chat_history = scrolledtext.ScrolledText(
            chat_frame, 
            height=20, 
            width=80,
            bg='#1a1a1a', 
            fg='#ffffff',
            insertbackground='#ffffff'
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Input frame
        input_frame = ttk.Frame(chat_frame, style='Dark.TFrame')
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Model selection
        ttk.Label(input_frame, text="Model:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="4b")
        model_combo = ttk.Combobox(input_frame, textvariable=self.model_var, 
                                  values=["4b", "27b"], width=10)
        model_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        # Message input
        ttk.Label(input_frame, text="Message:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.message_entry = tk.Entry(input_frame, width=50, bg='#404040', fg='#ffffff')
        self.message_entry.pack(side=tk.LEFT, padx=(5, 10), fill=tk.X, expand=True)
        self.message_entry.bind('<Return>', self.send_message)
        
        # Send button
        send_btn = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_btn.pack(side=tk.RIGHT)
        
        # Settings frame
        settings_frame = ttk.Frame(chat_frame, style='Dark.TFrame')
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Temperature and max tokens
        ttk.Label(settings_frame, text="Temperature:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.temp_var = tk.DoubleVar(value=0.7)
        temp_scale = tk.Scale(settings_frame, from_=0.1, to=1.0, resolution=0.1,
                             orient=tk.HORIZONTAL, variable=self.temp_var, length=100,
                             bg='#404040', fg='#ffffff')
        temp_scale.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(settings_frame, text="Max Tokens:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.tokens_var = tk.IntVar(value=50)
        tokens_scale = tk.Scale(settings_frame, from_=10, to=200, resolution=10,
                               orient=tk.HORIZONTAL, variable=self.tokens_var, length=100,
                               bg='#404040', fg='#ffffff')
        tokens_scale.pack(side=tk.LEFT, padx=5)
    
    def create_models_tab(self):
        """Create model management tab"""
        models_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(models_frame, text="🤖 Models")
        
        # Model status frame
        status_frame = ttk.Frame(models_frame, style='Dark.TFrame')
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(status_frame, text="Model Status:", style='Header.TLabel').pack(anchor=tk.W)
        
        # Model list
        columns = ('Model', 'Status', 'Memory', 'Target TPS', 'Actions')
        self.model_tree = ttk.Treeview(models_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.model_tree.heading(col, text=col)
            self.model_tree.column(col, width=120)
        
        self.model_tree.pack(fill=tk.X, padx=10, pady=10)
        
        # Model control buttons
        control_frame = ttk.Frame(models_frame, style='Dark.TFrame')
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Load 4B Model", 
                  command=lambda: self.load_model("4b")).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load 27B Model", 
                  command=lambda: self.load_model("27b")).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Unload 4B", 
                  command=lambda: self.unload_model("4b")).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Unload 27B", 
                  command=lambda: self.unload_model("27b")).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh Status", 
                  command=self.refresh_model_status).pack(side=tk.LEFT, padx=5)
        
        # Model download section
        download_frame = ttk.Frame(models_frame, style='Dark.TFrame')
        download_frame.pack(fill=tk.X, padx=10, pady=20)
        
        ttk.Label(download_frame, text="Model Downloads:", style='Header.TLabel').pack(anchor=tk.W)
        
        # Available models list
        available_models = [
            "Gemma 3 2B - 2.6GB",
            "Gemma 3 9B - 9.8GB", 
            "Llama 3.1 8B - 8.1GB",
            "Qwen 2.5 7B - 7.6GB",
            "CodeLlama 7B - 7.8GB"
        ]
        
        self.download_listbox = tk.Listbox(download_frame, height=6, 
                                         bg='#404040', fg='#ffffff')
        for model in available_models:
            self.download_listbox.insert(tk.END, model)
        self.download_listbox.pack(fill=tk.X, pady=5)
        
        download_btn_frame = ttk.Frame(download_frame, style='Dark.TFrame')
        download_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(download_btn_frame, text="Download Selected", 
                  command=self.download_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(download_btn_frame, text="Quantize & Optimize", 
                  command=self.quantize_model).pack(side=tk.LEFT, padx=5)
    
    def create_performance_tab(self):
        """Create performance monitoring tab"""
        perf_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(perf_frame, text="📊 Performance")
        
        # Performance metrics
        metrics_frame = ttk.Frame(perf_frame, style='Dark.TFrame')
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(metrics_frame, text="Real-Time Performance:", style='Header.TLabel').pack(anchor=tk.W)
        
        # Create matplotlib figure for performance plots
        self.perf_fig, (self.tps_ax, self.latency_ax) = plt.subplots(2, 1, figsize=(10, 6))
        self.perf_fig.patch.set_facecolor('#2d2d2d')
        
        for ax in [self.tps_ax, self.latency_ax]:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        self.tps_ax.set_title('Tokens Per Second', color='white')
        self.tps_ax.set_ylabel('TPS', color='white')
        
        self.latency_ax.set_title('Inference Latency', color='white')
        self.latency_ax.set_ylabel('Latency (ms)', color='white')
        self.latency_ax.set_xlabel('Time', color='white')
        
        canvas = FigureCanvasTkAgg(self.perf_fig, perf_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10)
        
        # Performance stats
        stats_frame = ttk.Frame(perf_frame, style='Dark.TFrame')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.perf_stats_text = scrolledtext.ScrolledText(
            stats_frame, height=8, bg='#1a1a1a', fg='#ffffff'
        )
        self.perf_stats_text.pack(fill=tk.X)
    
    def create_system_tab(self):
        """Create system monitoring tab"""
        system_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(system_frame, text="⚙️ System")
        
        # Hardware status
        hw_frame = ttk.Frame(system_frame, style='Dark.TFrame')
        hw_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(hw_frame, text="Hardware Status:", style='Header.TLabel').pack(anchor=tk.W)
        
        self.hw_status_text = scrolledtext.ScrolledText(
            hw_frame, height=10, bg='#1a1a1a', fg='#ffffff'
        )
        self.hw_status_text.pack(fill=tk.X, pady=5)
        
        # System metrics
        metrics_frame = ttk.Frame(system_frame, style='Dark.TFrame')
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(metrics_frame, text="System Metrics:", style='Header.TLabel').pack(anchor=tk.W)
        
        # Create system metrics plot
        self.sys_fig, (self.cpu_ax, self.mem_ax) = plt.subplots(2, 1, figsize=(10, 6))
        self.sys_fig.patch.set_facecolor('#2d2d2d')
        
        for ax in [self.cpu_ax, self.mem_ax]:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        self.cpu_ax.set_title('CPU Usage', color='white')
        self.cpu_ax.set_ylabel('CPU %', color='white')
        
        self.mem_ax.set_title('Memory Usage', color='white')
        self.mem_ax.set_ylabel('Memory %', color='white')
        self.mem_ax.set_xlabel('Time', color='white')
        
        sys_canvas = FigureCanvasTkAgg(self.sys_fig, metrics_frame)
        sys_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def send_message(self, event=None):
        """Send chat message to API"""
        message = self.message_entry.get().strip()
        if not message:
            return
        
        # Clear input
        self.message_entry.delete(0, tk.END)
        
        # Add to chat history
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_history.insert(tk.END, f"[{timestamp}] You: {message}\n")
        self.chat_history.see(tk.END)
        
        # Send request in background
        threading.Thread(target=self._send_chat_request, args=(message,), daemon=True).start()
    
    def _send_chat_request(self, message):
        """Send chat request to API"""
        try:
            payload = {
                "message": message,
                "model": self.model_var.get(),
                "max_tokens": self.tokens_var.get(),
                "temperature": self.temp_var.get()
            }
            
            start_time = time.time()
            response = requests.post(f"{self.api_base}/chat", json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Update chat history
            timestamp = datetime.now().strftime("%H:%M:%S")
            tps = data.get("tokens_per_second", 0)
            model = data.get("model", "Unknown")
            
            self.chat_history.insert(tk.END, 
                f"[{timestamp}] {model} ({tps:.1f} TPS): {data['response']}\n\n")
            self.chat_history.see(tk.END)
            
            # Update performance tracking
            model_type = self.model_var.get()
            self.performance_history[model_type].append({
                "time": time.time(),
                "tps": tps,
                "latency": data.get("time_taken", 0) * 1000,
                "tokens": data.get("tokens_generated", 0)
            })
            
            # Keep only last 50 entries
            if len(self.performance_history[model_type]) > 50:
                self.performance_history[model_type] = self.performance_history[model_type][-50:]
            
        except Exception as e:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.chat_history.insert(tk.END, f"[{timestamp}] Error: {str(e)}\n\n")
            self.chat_history.see(tk.END)
    
    def load_model(self, model_type):
        """Load a model"""
        try:
            response = requests.post(f"{self.api_base}/models/{model_type}/load", timeout=30)
            response.raise_for_status()
            data = response.json()
            messagebox.showinfo("Success", f"Model {model_type} loaded successfully")
            self.refresh_model_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model {model_type}: {str(e)}")
    
    def unload_model(self, model_type):
        """Unload a model"""
        try:
            response = requests.post(f"{self.api_base}/models/{model_type}/unload", timeout=10)
            response.raise_for_status()
            messagebox.showinfo("Success", f"Model {model_type} unloaded")
            self.refresh_model_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to unload model {model_type}: {str(e)}")
    
    def refresh_model_status(self):
        """Refresh model status display"""
        try:
            response = requests.get(f"{self.api_base}/models", timeout=10)
            response.raise_for_status()
            models = response.json()
            
            # Clear existing items
            for item in self.model_tree.get_children():
                self.model_tree.delete(item)
            
            # Add model info
            for model in models:
                status = "✅ Loaded" if model["loaded"] else "❌ Not Loaded"
                memory = f"{model['memory_mb']} MB"
                target_tps = f"{model['target_tps']} TPS"
                
                self.model_tree.insert('', tk.END, values=(
                    model["name"],
                    status,
                    memory,
                    target_tps,
                    "Load/Unload"
                ))
                
        except Exception as e:
            print(f"Failed to refresh model status: {e}")
    
    def download_model(self):
        """Simulate model download"""
        selection = self.download_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to download")
            return
        
        model_name = self.download_listbox.get(selection[0])
        
        # Simulate download with progress
        result = messagebox.askyesno("Confirm Download", 
                                   f"Download {model_name}?\n\nThis will download and quantize the model.")
        if result:
            messagebox.showinfo("Download Started", 
                              f"Download started for {model_name}\nCheck console for progress...")
    
    def quantize_model(self):
        """Simulate model quantization"""
        selection = self.download_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to quantize")
            return
        
        model_name = self.download_listbox.get(selection[0])
        messagebox.showinfo("Quantization", 
                          f"Starting quantization for {model_name}\n"
                          f"This will optimize the model for NPU+iGPU execution...")
    
    def update_performance_plots(self):
        """Update performance plots"""
        # Clear plots
        self.tps_ax.clear()
        self.latency_ax.clear()
        
        # Plot data for each model
        colors = {"4b": "#00ff88", "27b": "#ff8800"}
        
        for model_type, data in self.performance_history.items():
            if len(data) > 1:
                times = [d["time"] - data[0]["time"] for d in data]
                tps_values = [d["tps"] for d in data]
                latency_values = [d["latency"] for d in data]
                
                self.tps_ax.plot(times, tps_values, color=colors[model_type], 
                               label=f"Gemma 3 {model_type.upper()}", linewidth=2)
                self.latency_ax.plot(times, latency_values, color=colors[model_type], 
                                   label=f"Gemma 3 {model_type.upper()}", linewidth=2)
        
        # Style plots
        for ax in [self.tps_ax, self.latency_ax]:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        self.tps_ax.set_title('Tokens Per Second', color='white')
        self.tps_ax.set_ylabel('TPS', color='white')
        
        self.latency_ax.set_title('Inference Latency', color='white')
        self.latency_ax.set_ylabel('Latency (ms)', color='white')
        self.latency_ax.set_xlabel('Time (s)', color='white')
        
        self.perf_fig.tight_layout()
        self.perf_fig.canvas.draw()
    
    def update_system_plots(self):
        """Update system monitoring plots"""
        if len(self.system_metrics["time"]) > 1:
            times = self.system_metrics["time"]
            cpu_data = self.system_metrics["cpu"]
            mem_data = self.system_metrics["memory"]
            
            self.cpu_ax.clear()
            self.mem_ax.clear()
            
            self.cpu_ax.plot(times, cpu_data, color='#00ff88', linewidth=2)
            self.mem_ax.plot(times, mem_data, color='#ff8800', linewidth=2)
            
            for ax in [self.cpu_ax, self.mem_ax]:
                ax.set_facecolor('#1a1a1a')
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
            
            self.cpu_ax.set_title('CPU Usage', color='white')
            self.cpu_ax.set_ylabel('CPU %', color='white')
            self.cpu_ax.set_ylim(0, 100)
            
            self.mem_ax.set_title('Memory Usage', color='white')
            self.mem_ax.set_ylabel('Memory %', color='white')
            self.mem_ax.set_xlabel('Time (s)', color='white')
            self.mem_ax.set_ylim(0, 100)
            
            self.sys_fig.tight_layout()
            self.sys_fig.canvas.draw()
    
    def update_system_status(self):
        """Update system status display"""
        try:
            # Get API metrics
            response = requests.get(f"{self.api_base}/metrics", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                status_text = f"""🦄 UNICORN EXECUTION ENGINE STATUS
{'='*50}

🔧 HARDWARE:
   NPU: {'✅ Available' if data['hardware']['npu_available'] else '❌ Not Available'}
   iGPU: {'✅ Available' if data['hardware']['igpu_available'] else '❌ Not Available'}
   NPU Bandwidth: {data['hardware']['npu_memory_bandwidth']}
   CPU Cores: {data['hardware']['cpu_cores']}
   Total Memory: {data['hardware']['total_memory']}

📊 PERFORMANCE:
   4B Target TPS: {data['performance']['4b_target_tps']}
   27B Target TPS: {data['performance']['27b_target_tps']}
   Memory Optimization: {data['performance']['memory_optimization']}

⏱️  SERVER:
   Uptime: {data['server_uptime']:.1f}s
   Total Requests: {data['total_requests']}

🤖 MODELS:
"""
                
                for model_type, status in data['models_status'].items():
                    loaded = "✅ Loaded" if status['loaded'] else "❌ Not Loaded"
                    status_text += f"   {model_type.upper()}: {loaded}\n"
                
                self.hw_status_text.delete(1.0, tk.END)
                self.hw_status_text.insert(1.0, status_text)
        
        except Exception as e:
            error_text = f"❌ Connection Error: {str(e)}\n\nPlease ensure the API server is running."
            self.hw_status_text.delete(1.0, tk.END)
            self.hw_status_text.insert(1.0, error_text)
    
    def update_performance_stats(self):
        """Update performance statistics display"""
        stats_text = "🦄 PERFORMANCE STATISTICS\n"
        stats_text += "="*50 + "\n\n"
        
        for model_type, data in self.performance_history.items():
            if data:
                recent_data = data[-10:]  # Last 10 requests
                avg_tps = sum(d["tps"] for d in recent_data) / len(recent_data)
                avg_latency = sum(d["latency"] for d in recent_data) / len(recent_data)
                total_tokens = sum(d["tokens"] for d in recent_data)
                
                stats_text += f"📊 GEMMA 3 {model_type.upper()}:\n"
                stats_text += f"   Average TPS: {avg_tps:.2f}\n"
                stats_text += f"   Average Latency: {avg_latency:.1f}ms\n"
                stats_text += f"   Total Tokens (last 10): {total_tokens}\n"
                stats_text += f"   Requests: {len(data)}\n\n"
        
        # Add system performance
        if self.system_metrics["cpu"]:
            current_cpu = self.system_metrics["cpu"][-1]
            current_mem = self.system_metrics["memory"][-1]
            
            stats_text += f"⚙️  SYSTEM:\n"
            stats_text += f"   Current CPU: {current_cpu:.1f}%\n"
            stats_text += f"   Current Memory: {current_mem:.1f}%\n"
        
        self.perf_stats_text.delete(1.0, tk.END)
        self.perf_stats_text.insert(1.0, stats_text)
    
    def monitor_system(self):
        """Monitor system metrics"""
        current_time = time.time()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Add to metrics (keep last 100 points)
        self.system_metrics["time"].append(current_time)
        self.system_metrics["cpu"].append(cpu_percent)
        self.system_metrics["memory"].append(memory_percent)
        
        if len(self.system_metrics["time"]) > 100:
            for key in self.system_metrics:
                self.system_metrics[key] = self.system_metrics[key][-100:]
            
            # Adjust times to start from 0
            start_time = self.system_metrics["time"][0]
            self.system_metrics["time"] = [t - start_time for t in self.system_metrics["time"]]
    
    def start_monitoring(self):
        """Start background monitoring"""
        def monitor_loop():
            while True:
                try:
                    # Update system metrics
                    self.monitor_system()
                    
                    # Update GUI elements (schedule on main thread)
                    self.root.after(0, self.update_performance_plots)
                    self.root.after(0, self.update_system_plots)
                    self.root.after(0, self.update_system_status)
                    self.root.after(0, self.update_performance_stats)
                    self.root.after(0, self.refresh_model_status)
                    
                    time.sleep(2)  # Update every 2 seconds
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(5)
        
        # Start monitoring thread
        threading.Thread(target=monitor_loop, daemon=True).start()
    
    def run(self):
        """Run the GUI"""
        # Initial updates
        self.refresh_model_status()
        self.update_system_status()
        
        # Start main loop
        self.root.mainloop()

def main():
    """Main function"""
    print("🦄 Starting Unicorn Execution Engine GUI...")
    
    try:
        # Check if matplotlib is available
        import matplotlib
        matplotlib.use('TkAgg')
        
        # Start GUI
        gui = UnicornGUI()
        gui.run()
        
    except ImportError as e:
        print(f"❌ Required dependency missing: {e}")
        print("   Please install: sudo apt install python3-matplotlib python3-tk")
        
    except Exception as e:
        print(f"❌ GUI startup failed: {e}")

if __name__ == "__main__":
    main()