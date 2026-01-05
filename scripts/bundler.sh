#!/usr/bin/env bash
set -euo pipefail

BINARY_NAME="tiles"
DIST_DIR="dist"
SERVER_DIR="server"
TARGET="release"

VERSION=$(grep '^version' Cargo.toml | head -1 | awk -F'"' '{print $2}')
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
OUT_NAME="${BINARY_NAME}-v${VERSION}-${ARCH}-${OS}"

echo "🚀 Building ${BINARY_NAME} (${TARGET} mode)..."
cargo build --${TARGET}

mkdir -p "${DIST_DIR}/tmp"
cp "target/${TARGET}/${BINARY_NAME}" "${DIST_DIR}/tmp/"
cp -r "${SERVER_DIR}" "${DIST_DIR}/tmp/"

rm -rf "${DIST_DIR}/tmp/server/__pycache__"
rm -rf "${DIST_DIR}/tmp/server/.venv"

echo "📦 Creating ${OUT_NAME}.tar.gz..."
tar --exclude-from=scripts/tar.exclude -czf "${DIST_DIR}/${OUT_NAME}.tar.gz" -C "${DIST_DIR}/tmp" .

rm -rf "${DIST_DIR}/tmp"

echo "✅ Bundle created: ${DIST_DIR}/${OUT_NAME}.tar.gz"
