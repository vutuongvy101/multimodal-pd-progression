"""
Pytest configuration and shared fixtures for V1 tests

This file contains shared fixtures and configuration that can be used
across all test files.
"""

import pytest
import sys
from pathlib import Path

# Add V1_implementation to path for imports
v1_dir = Path(__file__).parent.parent
if str(v1_dir) not in sys.path:
    sys.path.insert(0, str(v1_dir))


@pytest.fixture
def v1_root():
    """Return the root directory of V1_implementation"""
    return Path(__file__).parent.parent


@pytest.fixture
def test_config():
    """Get default configuration for testing"""
    from training.config import get_default_config
    return get_default_config()


@pytest.fixture
def mock_base_dir(tmp_path):
    """Create a temporary directory for testing data paths"""
    return str(tmp_path / "test_data")


@pytest.fixture
def skip_if_no_data(test_config):
    """Skip test if data files are not available"""
    import os
    base_dir = test_config.data.base_dir
    if not os.path.exists(base_dir):
        pytest.skip(f"Data directory not found: {base_dir}")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring data files"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
