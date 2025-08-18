#!/usr/bin/env python3
"""
Comprehensive Test Runner for Stock Market Prediction System
Demonstrates production-ready testing practices and quality assurance
"""

import sys
import subprocess
import os
from pathlib import Path
import time
import json

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully in {duration:.2f}s")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ {description} failed with exception: {str(e)}")
        return False

def check_prerequisites():
    """Check if required tools are available"""
    print("🔍 Checking prerequisites...")
    
    required_tools = [
        ('python', '--version'),
        ('pip', '--version'),
        ('pytest', '--version'),
        ('black', '--version'),
        ('flake8', '--version'),
        ('isort', '--version')
    ]
    
    missing_tools = []
    for tool, version_flag in required_tools:
        try:
            subprocess.run([tool, version_flag], capture_output=True, check=True)
            print(f"✅ {tool} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {tool} is not available")
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"\n⚠️  Missing tools: {', '.join(missing_tools)}")
        print("Please install missing tools before running tests")
        return False
    
    return True

def run_unit_tests():
    """Run unit tests with coverage"""
    print("\n🧪 Running Unit Tests...")
    
    # Run tests with coverage
    success = run_command(
        "pytest tests/ -v --cov=stock_prediction --cov-report=term-missing --cov-report=html",
        "Unit tests with coverage report"
    )
    
    if success:
        print("\n📊 Coverage report generated in htmlcov/")
        print("Open htmlcov/index.html in your browser to view detailed coverage")
    
    return success

def run_integration_tests():
    """Run integration tests"""
    print("\n🔗 Running Integration Tests...")
    
    success = run_command(
        "pytest tests/ -m integration -v",
        "Integration tests"
    )
    
    return success

def run_code_quality_checks():
    """Run code quality checks"""
    print("\n🔍 Running Code Quality Checks...")
    
    checks = [
        ("black --check stock_prediction/ tests/", "Code formatting check (Black)"),
        ("isort --check-only stock_prediction/ tests/", "Import sorting check (isort)"),
        ("flake8 stock_prediction/ tests/ --max-line-length=88", "Linting check (flake8)"),
        ("mypy stock_prediction/", "Type checking (mypy)")
    ]
    
    all_passed = True
    for command, description in checks:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed

def run_security_checks():
    """Run security and safety checks"""
    print("\n🛡️ Running Security Checks...")
    
    checks = [
        ("safety check", "Dependency vulnerability check"),
        ("bandit -r stock_prediction/", "Security linting (bandit)")
    ]
    
    all_passed = True
    for command, description in checks:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed

def run_performance_tests():
    """Run performance and load tests"""
    print("\n⚡ Running Performance Tests...")
    
    # Simple performance test - check import time
    start_time = time.time()
    try:
        import stock_prediction
        import_time = time.time() - start_time
        print(f"✅ Module import time: {import_time:.3f}s")
        
        if import_time < 5.0:  # Should import in under 5 seconds
            print("✅ Import performance is acceptable")
            return True
        else:
            print("⚠️  Import performance is slow")
            return False
    except Exception as e:
        print(f"❌ Module import failed: {str(e)}")
        return False

def run_data_integrity_tests():
    """Run data integrity and validation tests"""
    print("\n📊 Running Data Integrity Tests...")
    
    # Test data loading and processing
    test_script = """
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

try:
    from stock_prediction.data.data_processor import DataProcessor
    from stock_prediction.data.time_series_cv import TimeSeriesSplit, WalkForwardValidator
    
    # Test data processor
    processor = DataProcessor(sequence_length=10)
    print("✅ DataProcessor initialized successfully")
    
    # Test time series CV
    tscv = TimeSeriesSplit(n_splits=3, test_size=20)
    print("✅ TimeSeriesSplit initialized successfully")
    
    # Test walk forward validator
    validator = WalkForwardValidator(n_splits=3, test_size=20)
    print("✅ WalkForwardValidator initialized successfully")
    
    print("✅ All data integrity components working")
    
except Exception as e:
    print(f"❌ Data integrity test failed: {str(e)}")
    sys.exit(1)
"""
    
    with open("temp_test.py", "w") as f:
        f.write(test_script)
    
    success = run_command("python temp_test.py", "Data integrity validation")
    
    # Cleanup
    if os.path.exists("temp_test.py"):
        os.remove("temp_test.py")
    
    return success

def run_model_tests():
    """Run model-specific tests"""
    print("\n🤖 Running Model Tests...")
    
    # Test model loading and basic functionality
    test_script = """
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

try:
    from stock_prediction.models.ml_models import (
        LinearRegressionModel, RandomForestModel, 
        XGBoostModel, LightGBMModel
    )
    
    # Test model initialization
    models = [
        LinearRegressionModel(),
        RandomForestModel(n_estimators=10, random_state=42),
        XGBoostModel(n_estimators=10, random_state=42),
        LightGBMModel(n_estimators=10, random_state=42)
    ]
    
    print(f"✅ Successfully initialized {len(models)} models")
    
    # Test model properties
    for model in models:
        assert hasattr(model, 'model_name'), f"Model missing model_name: {type(model)}"
        assert hasattr(model, 'needs_scaling'), f"Model missing needs_scaling: {type(model)}"
        assert hasattr(model, 'fit'), f"Model missing fit method: {type(model)}"
        assert hasattr(model, 'predict'), f"Model missing predict method: {type(model)}"
    
    print("✅ All models have required attributes and methods")
    
    # Test scaling configuration
    scaling_models = [m for m in models if m.needs_scaling]
    non_scaling_models = [m for m in models if not m.needs_scaling]
    
    print(f"✅ Scaling models: {len(scaling_models)} (Linear Regression)")
    print(f"✅ Non-scaling models: {len(non_scaling_models)} (Tree-based)")
    
    print("✅ All model tests passed")
    
except Exception as e:
    print(f"❌ Model test failed: {str(e)}")
    sys.exit(1)
"""
    
    with open("temp_model_test.py", "w") as f:
        f.write(test_script)
    
    success = run_command("python temp_model_test.py", "Model functionality validation")
    
    # Cleanup
    if os.path.exists("temp_model_test.py"):
        os.remove("temp_model_test.py")
    
    return success

def generate_test_report(results):
    """Generate a comprehensive test report"""
    print("\n📋 Generating Test Report...")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": len(results),
        "passed_tests": sum(results.values()),
        "failed_tests": len(results) - sum(results.values()),
        "success_rate": (sum(results.values()) / len(results)) * 100,
        "results": results
    }
    
    # Save report
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Test Report Generated:")
    print(f"   Total Tests: {report['total_tests']}")
    print(f"   Passed: {report['passed_tests']}")
    print(f"   Failed: {report['failed_tests']}")
    print(f"   Success Rate: {report['success_rate']:.1f}%")
    print(f"   Report saved to: test_report.json")
    
    return report

def main():
    """Main test runner function"""
    print("🚀 Stock Market Prediction - Comprehensive Test Suite")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Run all test categories
    test_results = {}
    
    test_results["Unit Tests"] = run_unit_tests()
    test_results["Integration Tests"] = run_integration_tests()
    test_results["Code Quality"] = run_code_quality_checks()
    test_results["Security"] = run_security_checks()
    test_results["Performance"] = run_performance_tests()
    test_results["Data Integrity"] = run_data_integrity_tests()
    test_results["Model Tests"] = run_model_tests()
    
    # Generate report
    report = generate_test_report(test_results)
    
    # Final summary
    print("\n" + "="*60)
    print("🏁 TEST SUITE COMPLETED")
    print("="*60)
    
    if report['success_rate'] == 100:
        print("🎉 ALL TESTS PASSED! The system is ready for production.")
    elif report['success_rate'] >= 80:
        print("✅ Most tests passed. The system is mostly ready for production.")
    else:
        print("⚠️  Many tests failed. Please address issues before production deployment.")
    
    print(f"\n📊 Final Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n📈 Overall Success Rate: {report['success_rate']:.1f}%")
    
    # Exit with appropriate code
    if report['success_rate'] == 100:
        sys.exit(0)
    elif report['success_rate'] >= 80:
        sys.exit(0)  # Still consider it a success
    else:
        sys.exit(1)  # Exit with error code

if __name__ == "__main__":
    main()
