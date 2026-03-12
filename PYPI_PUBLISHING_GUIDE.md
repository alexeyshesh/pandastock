# PyPI Publishing Guide for Pandastock

This guide will help you publish your pandastock package to PyPI so the README is properly displayed.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **API Token**: Generate an API token for publishing:
   - Go to https://pypi.org/manage/account/token/
   - Create a new token with "Entire account" scope
   - Save the token securely (you'll only see it once!)

## Build Your Package

First, install the required build tools:

```bash
pip install build twine
```

Then build your package:

```bash
python -m build
```

This will create a `dist/` directory with your built package files.

## Test Your Package Locally

Before publishing, you can test your package locally:

```bash
pip install dist/pandastock-0.0.3.3-py3-none-any.whl
```

Or test with TestPyPI (recommended):

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ pandastock
```

## Publish to PyPI

### Option 1: Using API Token (Recommended)

1. Set up your API token as an environment variable:
   ```bash
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

2. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

### Option 2: Using Username/Password

```bash
python -m twine upload dist/*
# You'll be prompted for your PyPI username and password
```

## Verify Your Package

After publishing, verify your package at:
- https://pypi.org/project/pandastock/

Check that:
- ✅ The README is properly rendered
- ✅ All metadata is correct
- ✅ The package can be installed: `pip install pandastock`

## What We've Done for You

Your project is now optimized for PyPI:

1. **README.md**: Comprehensive documentation with examples
2. **pyproject.toml**:
   - Properly configured with `readme = "README.md"`
   - Added keywords for better discoverability
   - Added classifiers for PyPI categorization
   - Added Issues URL
3. **MANIFEST.in**: Ensures README and LICENSE are included in the distribution

## Important Notes

### Version Management

Before publishing, ensure your version in [`pyproject.toml`](pyproject.toml:7) is correct:
```toml
version = "0.0.3.3"
```

PyPI does not allow overwriting existing versions. If you need to make changes, increment the version number.

### Semantic Versioning

Consider using semantic versioning:
- `0.0.3.3` → `0.0.4.0` (bug fixes)
- `0.0.4.0` → `0.1.0.0` (new features, backward compatible)
- `0.1.0.0` → `1.0.0.0` (major changes, breaking changes)

### README Rendering

PyPI supports:
- ✅ Markdown (`.md` files)
- ✅ reStructuredText (`.rst` files)
- ❌ HTML (not recommended)

Your README.md will be rendered as Markdown on PyPI.

### Troubleshooting

**README not showing on PyPI?**
- Ensure `readme = "README.md"` is in your [`pyproject.toml`](pyproject.toml:9)
- Check that README.md is in your package root
- Verify MANIFEST.in includes the README

**Upload failed?**
- Check if the version already exists on PyPI
- Verify your API token is correct
- Ensure all dependencies are properly listed

**Package not installable?**
- Check that all Python files are included
- Verify the package structure is correct
- Test installation locally first

## Next Steps

After successful publication:

1. **Announce your package**: Share on social media, forums, etc.
2. **Add badges**: Add PyPI badges to your README:
   ```markdown
   [![PyPI version](https://badge.fury.io/py/pandastock.svg)](https://badge.fury.io/py/pandastock)
   ```
3. **Monitor downloads**: Check your package statistics on PyPI
4. **Respond to issues**: Monitor the Issues section for user feedback

## Useful Commands

```bash
# Check what will be included in the package
python -m build --help

# Upload without checking (not recommended)
python -m twine upload --skip-existing dist/*

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Check package metadata
python -m twine check dist/*
```

## Additional Resources

- [PyPI Packaging Tutorial](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [PyPI Upload API](https://docs.pypi.org/api/)
