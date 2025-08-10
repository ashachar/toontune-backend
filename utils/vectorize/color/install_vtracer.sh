#!/bin/bash

# Installation script for vtracer
# Supports macOS, Linux, and provides Windows instructions

set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================"
echo "vtracer Installation Script"
echo "================================"
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"
echo ""

# Check if vtracer is already installed
if command -v vtracer &> /dev/null; then
    echo -e "${GREEN}✓ vtracer is already installed!${NC}"
    vtracer --version
    exit 0
fi

# Installation based on OS
case $OS in
    macos)
        echo "Installing vtracer on macOS..."
        echo ""
        
        # Try with cargo first (most reliable)
        if command -v cargo &> /dev/null; then
            echo -e "${YELLOW}Found cargo. Installing vtracer...${NC}"
            cargo install vtracer
            echo -e "${GREEN}✓ vtracer installed successfully via cargo!${NC}"
        else
            echo -e "${YELLOW}Cargo not found. Installing Rust first...${NC}"
            echo ""
            
            # Install Rust
            echo "Installing Rust toolchain..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            
            # Source cargo
            source "$HOME/.cargo/env"
            
            # Install vtracer
            echo ""
            echo "Installing vtracer..."
            cargo install vtracer
            
            echo ""
            echo -e "${GREEN}✓ vtracer installed successfully!${NC}"
            echo ""
            echo -e "${YELLOW}Note: You may need to restart your terminal or run:${NC}"
            echo "  source \$HOME/.cargo/env"
        fi
        ;;
        
    linux)
        echo "Installing vtracer on Linux..."
        echo ""
        
        # Check for cargo
        if command -v cargo &> /dev/null; then
            echo -e "${YELLOW}Found cargo. Installing vtracer...${NC}"
            cargo install vtracer
            echo -e "${GREEN}✓ vtracer installed successfully via cargo!${NC}"
        else
            echo -e "${YELLOW}Cargo not found. Installing Rust first...${NC}"
            echo ""
            
            # Install dependencies if needed
            if command -v apt-get &> /dev/null; then
                echo "Installing build dependencies..."
                sudo apt-get update
                sudo apt-get install -y build-essential curl
            elif command -v yum &> /dev/null; then
                echo "Installing build dependencies..."
                sudo yum groupinstall -y 'Development Tools'
                sudo yum install -y curl
            fi
            
            # Install Rust
            echo "Installing Rust toolchain..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            
            # Source cargo
            source "$HOME/.cargo/env"
            
            # Install vtracer
            echo ""
            echo "Installing vtracer..."
            cargo install vtracer
            
            echo ""
            echo -e "${GREEN}✓ vtracer installed successfully!${NC}"
            echo ""
            echo -e "${YELLOW}Note: You may need to restart your terminal or run:${NC}"
            echo "  source \$HOME/.cargo/env"
        fi
        ;;
        
    windows)
        echo "Windows detected. Please follow these steps:"
        echo ""
        echo "Option 1: Using Cargo (Recommended)"
        echo "  1. Install Rust from: https://www.rust-lang.org/tools/install"
        echo "  2. Open a new PowerShell or Command Prompt"
        echo "  3. Run: cargo install vtracer"
        echo ""
        echo "Option 2: Download Prebuilt Binary"
        echo "  1. Visit: https://github.com/visioncortex/vtracer/releases"
        echo "  2. Download the Windows binary (vtracer-windows-x64.zip)"
        echo "  3. Extract and add to your PATH"
        echo ""
        ;;
        
    *)
        echo -e "${RED}Unknown operating system.${NC}"
        echo ""
        echo "Manual installation instructions:"
        echo ""
        echo "1. Install Rust:"
        echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo ""
        echo "2. Install vtracer:"
        echo "   cargo install vtracer"
        echo ""
        echo "Or download binaries from:"
        echo "   https://github.com/visioncortex/vtracer/releases"
        exit 1
        ;;
esac

# Final check
echo ""
echo "Verifying installation..."
if command -v vtracer &> /dev/null; then
    echo -e "${GREEN}✓ vtracer installed successfully!${NC}"
    vtracer --version
else
    echo -e "${YELLOW}vtracer installed but not in PATH yet.${NC}"
    echo "Please restart your terminal or run:"
    echo "  source \$HOME/.cargo/env"
fi

echo ""
echo "================================"
echo "Installation complete!"
echo "================================"