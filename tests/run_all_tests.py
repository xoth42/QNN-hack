"""
Master Test Runner - Execute All Tests in One File

This script runs ALL test files in the correct order and provides
a comprehensive summary of the entire test suite.

Test Categories:
1. Quick Verification - Fast sanity checks
2. Component Tests - Individual component validation
3. Integration Tests - Full system integration
4. Performance Tests - Benchmarking

Usage:
    python run_all_tests.py
"""

import sys
import io
import subprocess
import time
from typing import List, Tuple

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class TestRunner:
    """Orchestrates execution of all test files."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    def run_test_file(self, test_file: str, description: str) -> bool:
        """Run a single test file and return success status."""
        print(f"\n{'='*80}")
        print(f"Running: {test_file}")
        print(f"Description: {description}")
        print(f"{'='*80}")
        
        try:
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Print output
            if result.stdout:
                print(result.stdout)
            
            # Check for errors
            if result.returncode == 0:
                print(f"\n[PASS] {test_file}")
                self.results.append((test_file, "PASS", None))
                return True
            else:
                print(f"\n[FAIL] {test_file} (exit code {result.returncode})")
                if result.stderr:
                    print("Error output:")
                    print(result.stderr)
                self.results.append((test_file, "FAIL", f"Exit code {result.returncode}"))
                return False
                
        except subprocess.TimeoutExpired:
            print(f"\n[FAIL] {test_file} (timeout)")
            self.results.append((test_file, "FAIL", "Timeout"))
            return False
        except Exception as e:
            print(f"\n[FAIL] {test_file} (exception: {e})")
            self.results.append((test_file, "FAIL", str(e)))
            return False
    
    def print_summary(self):
        """Print comprehensive test summary."""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("MASTER TEST SUITE SUMMARY")
        print("="*80)
        
        passed = sum(1 for _, status, _ in self.results if status == "PASS")
        failed = sum(1 for _, status, _ in self.results if status == "FAIL")
        total = len(self.results)
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        print(f"Total Time: {elapsed_time:.1f}s")
        
        if failed > 0:
            print("\n" + "="*80)
            print("FAILED TESTS:")
            print("="*80)
            for test_file, status, error in self.results:
                if status == "FAIL":
                    print(f"\n[FAIL] {test_file}")
                    if error:
                        print(f"  Error: {error}")
        
        print("\n" + "="*80)
        if failed == 0:
            print("SUCCESS - ALL TESTS PASSED!")
            print("System is PRODUCTION READY")
        else:
            print(f"WARNING - {failed} test(s) failed")
            print("Review errors above")
        print("="*80)
        
        return failed == 0


def main():
    """Run all tests in order."""
    print("="*80)
    print("MASTER TEST RUNNER")
    print("Running ALL tests in the test suite")
    print("="*80)
    
    runner = TestRunner()
    
    # Define all tests in execution order
    # Note: All test files are in the tests/ folder
    tests = [
        # Quick verification tests
        ("test_quick_verify.py", "Quick verification - 10 component tests"),
        
        # Core component tests
        ("test_unitary_decomposition.py", "Unitary decomposition validation"),
        ("test_walsh_validation.py", "Walsh decomposition validation"),
        ("test_walsh_circuit_decomposition.py", "Walsh circuit decomposition test"),
        ("test_density_qnn.py", "Density QNN utilities test"),
        
        # Integration tests
        ("test_decomposition.py", "V @ D @ V† decomposition test"),
        ("test_simple_decomp.py", "Simple decomposition test"),
        ("test_apply_circuit.py", "Circuit application test"),
        
        # QNN tests
        ("test_qnn_forward_pass.py", "QNN forward pass test"),
        ("test_qnn_gradients.py", "QNN gradient flow test"),
        ("test_rbs_networks.py", "RBS network patterns test"),
        ("test_complete_qnn.py", "Complete QNN pipeline test"),
        
        # Final comprehensive test
        ("final_test.py", "Final comprehensive test - 12 tests"),
    ]
    
    print(f"\nTotal test files to run: {len(tests)}")
    print("This may take a few minutes...\n")
    
    # Run all tests
    for test_file, description in tests:
        runner.run_test_file(test_file, description)
    
    # Print summary
    success = runner.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
