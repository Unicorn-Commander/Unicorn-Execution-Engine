#!/bin/bash
#
# Gemma 3n E4B Unicorn Execution Engine Start Script
# Comprehensive startup script for the complete Gemma 3n E4B pipeline
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine"
AI_ENV_SCRIPT="/home/ucadmin/activate-uc1-ai-py311.sh"
MODEL_PATH="./models/gemma-3n-e4b-it"
DEFAULT_PORT=8000
DEFAULT_HOST="0.0.0.0"
LOG_LEVEL="info"
WORKERS=1

# Default mode
MODE="api"

# Help function
show_help() {
    echo -e "${CYAN}🦄 Gemma 3n E4B Unicorn Execution Engine${NC}"
    echo -e "${CYAN}=========================================${NC}"
    echo ""
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Available Modes:${NC}"
    echo "  api         Start OpenAI API server (default)"
    echo "  test        Run pipeline tests"
    echo "  chat        Start terminal chat interface"
    echo "  benchmark   Run performance benchmarks"
    echo "  status      Check system status"
    echo ""
    echo -e "${YELLOW}API Server Options:${NC}"
    echo "  --port PORT        Server port (default: $DEFAULT_PORT)"
    echo "  --host HOST        Server host (default: $DEFAULT_HOST)"
    echo "  --workers NUM      Number of workers (default: $WORKERS)"
    echo "  --model-path PATH  Model path (default: $MODEL_PATH)"
    echo "  --log-level LEVEL  Log level (default: $LOG_LEVEL)"
    echo ""
    echo -e "${YELLOW}General Options:${NC}"
    echo "  --help, -h         Show this help"
    echo "  --verbose, -v      Verbose output"
    echo "  --no-gpu          Disable GPU acceleration"
    echo "  --no-npu          Disable NPU acceleration"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 api                           # Start API server on default port"
    echo "  $0 api --port 8001              # Start API server on port 8001"
    echo "  $0 test                         # Run pipeline tests"
    echo "  $0 chat                         # Start terminal chat"
    echo "  $0 benchmark                    # Run benchmarks"
    echo "  $0 status                       # Check system status"
    echo ""
}

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            api|test|chat|benchmark|status)
                MODE="$1"
                shift
                ;;
            --port)
                DEFAULT_PORT="$2"
                shift 2
                ;;
            --host)
                DEFAULT_HOST="$2"
                shift 2
                ;;
            --workers)
                WORKERS="$2"
                shift 2
                ;;
            --model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            --log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            --verbose|-v)
                set -x
                shift
                ;;
            --no-gpu)
                export DISABLE_GPU=1
                shift
                ;;
            --no-npu)
                export DISABLE_NPU=1
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -d "$PROJECT_DIR" ]]; then
        log_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    
    # Change to project directory
    cd "$PROJECT_DIR"
    log_info "Working directory: $(pwd)"
    
    # Check AI environment script
    if [[ ! -f "$AI_ENV_SCRIPT" ]]; then
        log_error "AI environment script not found: $AI_ENV_SCRIPT"
        log_error "Please ensure the AI environment is properly installed"
        exit 1
    fi
    
    # Check for required Python files
    local required_files=(
        "gemma3n_e4b_openai_api_server.py"
        "gemma3n_e4b_unicorn_loader.py"
        "test_gemma3n_e4b_core.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

# Activate AI environment
activate_environment() {
    log_step "Activating AI environment..."
    
    # Source the AI environment
    if source "$AI_ENV_SCRIPT"; then
        log_success "AI environment activated"
        log_info "Python version: $(python --version)"
        log_info "Environment: $VIRTUAL_ENV"
    else
        log_error "Failed to activate AI environment"
        exit 1
    fi
}

# Check hardware status
check_hardware() {
    log_step "Checking hardware status..."
    
    # Check NPU
    if command -v xrt-smi &> /dev/null; then
        log_info "NPU Status:"
        if xrt-smi examine 2>/dev/null | grep -qi "phoenix"; then
            log_success "✅ NPU Phoenix detected"
            
            # Enable turbo mode if not disabled
            if [[ -z "$DISABLE_NPU" ]]; then
                log_info "Enabling NPU turbo mode..."
                sudo xrt-smi configure --pmode turbo 2>/dev/null || log_warn "Could not enable turbo mode (requires sudo)"
            fi
        else
            log_warn "⚠️ NPU Phoenix not detected"
        fi
    else
        log_warn "⚠️ XRT tools not found"
    fi
    
    # Check iGPU
    if command -v vulkaninfo &> /dev/null; then
        log_info "iGPU Status:"
        if vulkaninfo --summary 2>/dev/null | grep -q "AMD Radeon"; then
            log_success "✅ AMD Radeon iGPU detected"
        else
            log_warn "⚠️ AMD Radeon iGPU not detected"
        fi
    else
        log_warn "⚠️ Vulkan tools not found"
    fi
    
    # Check ROCm
    if command -v rocm-smi &> /dev/null; then
        log_info "ROCm Status:"
        if rocm-smi --showuse 2>/dev/null | grep -q "GPU"; then
            log_success "✅ ROCm available"
        else
            log_warn "⚠️ ROCm not available"
        fi
    fi
    
    log_success "Hardware check completed"
}

# Start API server
start_api_server() {
    log_step "Starting Gemma 3n E4B OpenAI API Server..."
    
    log_info "Configuration:"
    log_info "  Host: $DEFAULT_HOST"
    log_info "  Port: $DEFAULT_PORT"
    log_info "  Workers: $WORKERS"
    log_info "  Model: $MODEL_PATH"
    log_info "  Log Level: $LOG_LEVEL"
    
    # Create logs directory
    mkdir -p logs
    
    # Start the server
    log_info "Starting server... (this may take a few seconds for initialization)"
    
    python gemma3n_e4b_openai_api_server.py \
        --host "$DEFAULT_HOST" \
        --port "$DEFAULT_PORT" \
        --workers "$WORKERS" \
        --model-path "$MODEL_PATH" \
        --log-level "$LOG_LEVEL" 2>&1 | tee "logs/gemma3n_e4b_server_$(date +%Y%m%d_%H%M%S).log"
}

# Run tests
run_tests() {
    log_step "Running Gemma 3n E4B pipeline tests..."
    
    python test_gemma3n_e4b_core.py
    
    if [[ $? -eq 0 ]]; then
        log_success "✅ All tests passed!"
    else
        log_error "❌ Some tests failed"
        exit 1
    fi
}

# Start chat interface
start_chat() {
    log_step "Starting Gemma 3n E4B terminal chat..."
    
    # Check if chat script exists, if not create a simple one
    if [[ ! -f "gemma3n_e4b_chat.py" ]]; then
        log_info "Creating chat interface..."
        cat > gemma3n_e4b_chat.py << 'EOF'
#!/usr/bin/env python3
"""
Simple chat interface for Gemma 3n E4B
"""
import sys
from gemma3n_e4b_unicorn_loader import (
    Gemma3nE4BUnicornLoader, ModelConfig, HardwareConfig, InferenceConfig, InferenceMode
)

def main():
    print("🦄 Gemma 3n E4B Chat Interface")
    print("=" * 40)
    
    # Initialize loader
    model_config = ModelConfig(
        model_path="./models/gemma-3n-e4b-it",
        elastic_enabled=True,
        quantization_enabled=True,
        mix_n_match_enabled=True
    )
    
    hardware_config = HardwareConfig(
        npu_enabled=True,
        igpu_enabled=True,
        hma_enabled=True,
        turbo_mode=True
    )
    
    print("Loading model...")
    loader = Gemma3nE4BUnicornLoader(model_config, hardware_config)
    
    if not loader.load_model():
        print("❌ Failed to load model")
        return 1
    
    print("✅ Model loaded! Type 'quit' to exit.")
    print()
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            inference_config = InferenceConfig(
                mode=InferenceMode.BALANCED,
                max_tokens=150,
                temperature=0.7
            )
            
            result = loader.generate(user_input, inference_config)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"Assistant: {result['generated_text']}")
                print(f"[{result['tokens_per_second']:.1f} TPS, {result['tokens_generated']} tokens]")
            print()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    finally:
        loader.shutdown()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
    fi
    
    python gemma3n_e4b_chat.py
}

# Run benchmarks
run_benchmark() {
    log_step "Running Gemma 3n E4B benchmarks..."
    
    # Check if benchmark script exists
    if [[ -f "gemma3n_e4b_benchmark.py" ]]; then
        python gemma3n_e4b_benchmark.py
    else
        log_info "Running core tests as benchmark..."
        python test_gemma3n_e4b_core.py
    fi
}

# Check system status
check_status() {
    log_step "Checking Gemma 3n E4B system status..."
    
    # Check if API server is running
    if pgrep -f "gemma3n_e4b_openai_api_server" > /dev/null; then
        log_success "✅ API server is running"
        
        # Check API health if curl is available
        if command -v curl &> /dev/null; then
            log_info "Checking API health..."
            if curl -s "http://localhost:$DEFAULT_PORT/health" > /dev/null; then
                log_success "✅ API server is healthy"
            else
                log_warn "⚠️ API server not responding"
            fi
        fi
    else
        log_info "ℹ️ API server is not running"
    fi
    
    # Check hardware again
    check_hardware
    
    # Show resource usage
    log_info "System Resources:"
    if command -v free &> /dev/null; then
        echo "Memory usage:"
        free -h | grep -E "Mem:|Swap:"
    fi
    
    if command -v df &> /dev/null; then
        echo "Disk usage:"
        df -h "$PROJECT_DIR" | tail -1
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Kill any background processes if needed
    jobs -p | xargs -r kill 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    echo -e "${PURPLE}"
    echo "🦄 Gemma 3n E4B Unicorn Execution Engine"
    echo "========================================"
    echo -e "${NC}"
    
    # Parse arguments
    parse_args "$@"
    
    # Check prerequisites
    check_prerequisites
    
    # Activate environment
    activate_environment
    
    # Check hardware
    check_hardware
    
    # Execute based on mode
    case $MODE in
        api)
            start_api_server
            ;;
        test)
            run_tests
            ;;
        chat)
            start_chat
            ;;
        benchmark)
            run_benchmark
            ;;
        status)
            check_status
            ;;
        *)
            log_error "Unknown mode: $MODE"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"