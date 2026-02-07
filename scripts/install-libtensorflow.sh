#!/bin/bash
# Install libtensorflow for SHGAT-TF FFI backend
#
# Usage: ./install-libtensorflow.sh [cpu|gpu]
#
# Requirements:
# - Linux x86_64 (Ubuntu/Debian)
# - sudo access
#
# For GPU version, also need CUDA 11.8+ installed

set -e

VERSION="2.15.0"
VARIANT="${1:-cpu}"

if [[ "$VARIANT" != "cpu" && "$VARIANT" != "gpu" ]]; then
    echo "Usage: $0 [cpu|gpu]"
    exit 1
fi

echo "[install-libtensorflow] Installing libtensorflow $VERSION ($VARIANT)..."

# Download
FILENAME="libtensorflow-${VARIANT}-linux-x86_64-${VERSION}.tar.gz"
URL="https://storage.googleapis.com/tensorflow/libtensorflow/${FILENAME}"

echo "[install-libtensorflow] Downloading from $URL..."
cd /tmp
wget -q --show-progress "$URL"

# Extract to /usr/local
echo "[install-libtensorflow] Extracting to /usr/local..."
sudo tar -C /usr/local -xzf "$FILENAME"

# Update library cache
echo "[install-libtensorflow] Running ldconfig..."
sudo ldconfig

# Verify installation
echo "[install-libtensorflow] Verifying installation..."
if [ -f /usr/local/lib/libtensorflow.so ]; then
    echo "[install-libtensorflow] SUCCESS: libtensorflow installed at /usr/local/lib/libtensorflow.so"
    ls -la /usr/local/lib/libtensorflow*
else
    echo "[install-libtensorflow] ERROR: Installation failed"
    exit 1
fi

# Cleanup
rm -f "/tmp/$FILENAME"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To verify in Deno:"
echo "  deno eval 'import tff from \"./lib/shgat-tf/src/tf/tf-ffi.ts\"; console.log(tff.isAvailable(), tff.version())'"
echo ""
