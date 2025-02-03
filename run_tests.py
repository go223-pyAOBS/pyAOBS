"""
Test runner for pyAOBS package

Run all tests with proper Python path setup
"""

import unittest
import sys
from pathlib import Path

# Add package root to Python path
package_root = Path(__file__).parent
sys.path.append(str(package_root))

# Discover and run tests
test_loader = unittest.TestLoader()
test_suite = test_loader.discover('pyAOBS', pattern='test_*.py')

runner = unittest.TextTestRunner(verbosity=2)
runner.run(test_suite) 