#!/bin/bash

# Pandastock PyPI Publishing Script
# This script helps you build and publish your package to PyPI

set -e  # Exit on error

echo "🚀 Pandastock PyPI Publishing Script"
echo "===================================="
echo ""

# Check if build tools are installed
echo "📦 Checking build tools..."
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed"
    exit 1
fi

if ! python -c "import build" 2>/dev/null; then
    echo "⚠️  Build tool not found. Installing..."
    pip install build twine
fi

echo "✅ Build tools ready"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
echo "✅ Cleaned"
echo ""

# Build the package
echo "🔨 Building package..."
python -m build
echo "✅ Build complete"
echo ""

# Check the package
echo "🔍 Checking package..."
python -m twine check dist/*
echo "✅ Package check passed"
echo ""

# Ask where to publish
echo "Where do you want to publish?"
echo "1) TestPyPI (for testing)"
echo "2) PyPI (production)"
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "📤 Uploading to TestPyPI..."
        python -m twine upload --repository testpypi dist/*
        echo ""
        echo "✅ Uploaded to TestPyPI!"
        echo ""
        echo "To test installation:"
        echo "  pip install --index-url https://test.pypi.org/simple/ pandastock"
        ;;
    2)
        echo ""
        echo "📤 Uploading to PyPI..."
        python -m twine upload dist/*
        echo ""
        echo "✅ Uploaded to PyPI!"
        echo ""
        echo "View your package at:"
        echo "  https://pypi.org/project/pandastock/"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Done!"
