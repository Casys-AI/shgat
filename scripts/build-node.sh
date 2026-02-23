#!/usr/bin/env bash
# Build shgat-tf for Node.js distribution
#
# What this does:
# 1. Copies src/ to dist-node/
# 2. Replaces backend.ts with backend.node.ts (tfjs-node instead of tfjs + WASM)
# 3. Strips Deno-specific "npm:" prefixes from imports
# 4. Removes .ts extensions from imports (Node ESM convention)
#
# Usage:
#   cd lib/shgat-tf && ./scripts/build-node.sh
#
# Output: dist-node/ ready for npm publish
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist-node"

echo "[build-node] Building Node.js distribution..."

# Clean
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Copy source files
cp -r "$ROOT_DIR/src" "$DIST_DIR/src"
cp "$ROOT_DIR/mod.ts" "$DIST_DIR/mod.ts"

# Replace backend.ts with backend.node.ts
cp "$DIST_DIR/src/tf/backend.node.ts" "$DIST_DIR/src/tf/backend.ts"
rm "$DIST_DIR/src/tf/backend.node.ts"

# Strip "npm:" prefix from any remaining imports (shouldn't be any, but just in case)
find "$DIST_DIR" -name "*.ts" -exec sed -i 's/from "npm:\(.*\)"/from "\1"/g' {} +

# Strip .ts extensions from relative imports for Node ESM
# Matches: from "./foo.ts" or from "../bar/baz.ts"
find "$DIST_DIR" -name "*.ts" -exec sed -i 's/from "\(\.\.[^"]*\)\.ts"/from "\1.js"/g' {} +

# Rename .ts → .js (optional: if you want to tsc compile instead, skip this)
# For now we keep .ts and expect consumers to use ts-node or tsx

# Generate package.json
cat > "$DIST_DIR/package.json" <<'PKGJSON'
{
  "name": "@casys/shgat",
  "version": "0.1.0",
  "description": "SuperHyperGraph Attention Networks with TensorFlow.js",
  "type": "module",
  "main": "mod.ts",
  "types": "mod.ts",
  "scripts": {
    "build": "tsc",
    "test": "tsx --test tests/"
  },
  "dependencies": {
    "@tensorflow/tfjs-node": "^4.22.0"
  },
  "devDependencies": {
    "typescript": "^5.4.0",
    "tsx": "^4.0.0"
  },
  "engines": {
    "node": ">=20.0.0"
  },
  "license": "MIT"
}
PKGJSON

echo "[build-node] Done! Output: $DIST_DIR"
echo ""
echo "Next steps:"
echo "  cd $DIST_DIR"
echo "  npm install"
echo "  npm test"
