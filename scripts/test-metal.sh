#!/usr/bin/env bash
# =============================================================================
# Metal Backend Test Script for llama-gguf
# =============================================================================
#
# This script tests the Metal backend on macOS hardware. It is designed to be
# run OUTSIDE of GitHub Actions/CI workflows, on real macOS hardware with a
# Metal-capable GPU (Apple Silicon or AMD discrete GPU).
#
# Usage:
#   ./scripts/test-metal.sh                    # Full test suite
#   ./scripts/test-metal.sh --unit-only        # Unit tests only
#   ./scripts/test-metal.sh --model <path>     # Test with specific GGUF model
#   ./scripts/test-metal.sh --download-model   # Auto-download a small test model
#
# Requirements:
#   - macOS 10.15+ with Metal-capable GPU
#   - Rust toolchain (rustup)
#   - Xcode Command Line Tools (xcode-select --install)
#   - ~500MB free disk space (for test model download)
#
# Recommended test environments:
#   - Developer's MacBook (Apple Silicon M1/M2/M3/M4)
#   - AWS EC2 Mac instances (mac2.metal, mac2-m2.metal, mac2-m4.metal)
#   - MacStadium bare metal Macs
#   - Cirrus CI with Tart (macOS VMs on Apple Silicon)
#
# NOTE: Metal APIs CANNOT be tested on Linux. There is no QEMU emulation,
# no translation layer, and the metal crate requires macOS to compile.
# This script must be run on actual macOS hardware.
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
TEST_MODEL_NAME="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
TEST_MODEL_DIR="$PROJECT_DIR/.test-models"
RESULTS_FILE="$PROJECT_DIR/.metal-test-results.json"

# Parse arguments
UNIT_ONLY=false
MODEL_PATH=""
DOWNLOAD_MODEL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --unit-only)
            UNIT_ONLY=true
            shift
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --download-model)
            DOWNLOAD_MODEL=true
            shift
            ;;
        --help|-h)
            head -35 "$0" | tail -30
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# =============================================================================
# Preflight checks
# =============================================================================

echo -e "${BLUE}=== llama-gguf Metal Backend Test Suite ===${NC}"
echo ""

# Check we're on macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo -e "${RED}ERROR: This script must be run on macOS.${NC}"
    echo ""
    echo "Metal APIs cannot be tested on Linux or Windows."
    echo "Run this on macOS hardware (MacBook, Mac Studio, AWS EC2 Mac, etc.)"
    exit 1
fi

# Check architecture
ARCH="$(uname -m)"
echo -e "Platform:     ${GREEN}macOS $(sw_vers -productVersion) ($ARCH)${NC}"

if [[ "$ARCH" == "arm64" ]]; then
    echo -e "Architecture: ${GREEN}Apple Silicon (unified memory)${NC}"
else
    echo -e "Architecture: ${YELLOW}Intel x86_64 (discrete GPU)${NC}"
fi

# Check Metal availability
if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
    GPU_NAME=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | sed 's/.*: //')
    echo -e "GPU:          ${GREEN}${GPU_NAME} (Metal supported)${NC}"
else
    echo -e "${RED}ERROR: No Metal-capable GPU found${NC}"
    exit 1
fi

# Check Xcode CLI tools
if ! xcrun --find metal &>/dev/null; then
    echo -e "${RED}ERROR: Xcode Command Line Tools not installed${NC}"
    echo "Run: xcode-select --install"
    exit 1
fi
echo -e "Metal SDK:    ${GREEN}$(xcrun --find metal)${NC}"

# Check Rust
if ! command -v cargo &>/dev/null; then
    echo -e "${RED}ERROR: Rust toolchain not found${NC}"
    echo "Install from: https://rustup.rs"
    exit 1
fi
echo -e "Rust:         ${GREEN}$(rustc --version)${NC}"

echo ""

# =============================================================================
# Phase 1: Build with Metal feature
# =============================================================================

echo -e "${BLUE}--- Phase 1: Building with Metal backend ---${NC}"
cd "$PROJECT_DIR"

# Build the library with Metal support
echo "Building llama-gguf with --features metal..."
BUILD_START=$(date +%s)

if cargo build --features metal 2>&1; then
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
    echo -e "${GREEN}Build succeeded in ${BUILD_TIME}s${NC}"
else
    echo -e "${RED}Build FAILED${NC}"
    echo ""
    echo "Common fixes:"
    echo "  1. Install Xcode CLI tools: xcode-select --install"
    echo "  2. Ensure macOS SDK is available: xcrun --show-sdk-path"
    echo "  3. Check Cargo.toml metal feature dependencies"
    exit 1
fi

echo ""

# =============================================================================
# Phase 2: Unit tests
# =============================================================================

echo -e "${BLUE}--- Phase 2: Metal Backend Unit Tests ---${NC}"

TEST_START=$(date +%s)
UNIT_PASSED=0
UNIT_FAILED=0

echo "Running Metal backend unit tests..."
if cargo test --features metal -- metal 2>&1 | tee /tmp/metal-test-output.txt; then
    UNIT_PASSED=$(grep -c "test result: ok" /tmp/metal-test-output.txt || echo "0")
    echo -e "${GREEN}Unit tests PASSED${NC}"
else
    UNIT_FAILED=1
    echo -e "${RED}Some unit tests FAILED${NC}"
fi

TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))
echo "Unit tests completed in ${TEST_TIME}s"

# Extract test counts
TOTAL_TESTS=$(grep "test result:" /tmp/metal-test-output.txt | tail -1 | grep -oP '\d+ passed' | grep -oP '\d+' || echo "0")
FAILED_TESTS=$(grep "test result:" /tmp/metal-test-output.txt | tail -1 | grep -oP '\d+ failed' | grep -oP '\d+' || echo "0")

echo "  Passed: $TOTAL_TESTS"
echo "  Failed: $FAILED_TESTS"

echo ""

if [[ "$UNIT_ONLY" == true ]]; then
    echo -e "${BLUE}=== Unit-only mode: skipping integration tests ===${NC}"
    if [[ "$UNIT_FAILED" -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Tests failed${NC}"
        exit 1
    fi
fi

# =============================================================================
# Phase 3: Model loading integration test
# =============================================================================

echo -e "${BLUE}--- Phase 3: GGUF Model Loading on Metal ---${NC}"

# Resolve model path
if [[ -z "$MODEL_PATH" ]]; then
    if [[ "$DOWNLOAD_MODEL" == true ]]; then
        mkdir -p "$TEST_MODEL_DIR"
        MODEL_PATH="$TEST_MODEL_DIR/$TEST_MODEL_NAME"

        if [[ ! -f "$MODEL_PATH" ]]; then
            echo "Downloading test model: $TEST_MODEL_NAME"
            echo "URL: $TEST_MODEL_URL"
            curl -L --progress-bar -o "$MODEL_PATH" "$TEST_MODEL_URL"
            echo -e "${GREEN}Download complete: $(du -h "$MODEL_PATH" | cut -f1)${NC}"
        else
            echo "Using cached model: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
        fi
    else
        echo -e "${YELLOW}No model specified. Use --download-model or --model <path>${NC}"
        echo "Skipping integration test."
        echo ""

        # Still report results
        if [[ "$UNIT_FAILED" -eq 0 ]]; then
            echo -e "${GREEN}Unit tests passed. Run with --download-model for full integration test.${NC}"
            exit 0
        else
            echo -e "${RED}Unit tests had failures${NC}"
            exit 1
        fi
    fi
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo -e "${RED}ERROR: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

echo "Model: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
echo ""

# Run integration test: load model with Metal backend and generate tokens
echo "Running model loading test..."

INTEGRATION_START=$(date +%s)

# Build and run the integration test
if cargo test --features metal --test metal_integration -- --nocapture 2>&1 | tee /tmp/metal-integration-output.txt; then
    INTEGRATION_RESULT="PASSED"
    echo -e "${GREEN}Integration test PASSED${NC}"
else
    INTEGRATION_RESULT="FAILED"
    echo -e "${RED}Integration test FAILED${NC}"
fi

INTEGRATION_END=$(date +%s)
INTEGRATION_TIME=$((INTEGRATION_END - INTEGRATION_START))
echo "Integration test completed in ${INTEGRATION_TIME}s"

echo ""

# =============================================================================
# Results Summary
# =============================================================================

echo -e "${BLUE}=== Test Results Summary ===${NC}"
echo ""
echo "Platform:           macOS $(sw_vers -productVersion) ($ARCH)"
echo "GPU:                $GPU_NAME"
echo "Build:              ${BUILD_TIME}s"
echo "Unit Tests:         $TOTAL_TESTS passed, $FAILED_TESTS failed (${TEST_TIME}s)"
echo "Integration Test:   $INTEGRATION_RESULT (${INTEGRATION_TIME}s)"
echo ""

# Write results JSON
cat > "$RESULTS_FILE" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "platform": {
    "os": "macOS",
    "os_version": "$(sw_vers -productVersion)",
    "arch": "$ARCH",
    "gpu": "$GPU_NAME"
  },
  "build": {
    "success": true,
    "time_seconds": $BUILD_TIME
  },
  "unit_tests": {
    "passed": $TOTAL_TESTS,
    "failed": $FAILED_TESTS,
    "time_seconds": $TEST_TIME
  },
  "integration": {
    "result": "$INTEGRATION_RESULT",
    "model": "$(basename "$MODEL_PATH" 2>/dev/null || echo "none")",
    "time_seconds": $INTEGRATION_TIME
  }
}
EOF

echo "Results written to: $RESULTS_FILE"

# Exit code
if [[ "$FAILED_TESTS" -eq 0 && "$INTEGRATION_RESULT" == "PASSED" ]]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
