"""Tests for Phase 1: Environment Setup."""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.phase1.verify_setup import (
    check_python_version,
    check_dependencies,
    check_directories,
    check_data_file,
)


class TestPythonVersion:
    """Test Python version check."""
    
    def test_python_version(self):
        """Test that Python version is 3.10+."""
        result = check_python_version()
        assert result is True, "Python 3.10+ is required"


class TestDependencies:
    """Test dependency checks."""
    
    def test_dependencies(self):
        """Test that required dependencies are installed."""
        result = check_dependencies()
        assert result is True, "All required dependencies must be installed"


class TestDirectories:
    """Test directory checks."""
    
    def test_directories_exist(self):
        """Test that required directories exist."""
        result = check_directories()
        assert result is True, "All required directories must exist"


class TestDataFile:
    """Test data file checks."""
    
    def test_data_file_check(self):
        """Test that data file check runs without error."""
        # This should not raise an exception
        result = check_data_file()
        # Result can be True or False, both are valid
        assert isinstance(result, bool)

