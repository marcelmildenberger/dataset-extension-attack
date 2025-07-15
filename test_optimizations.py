#!/usr/bin/env python3
"""
Test script to verify performance optimizations are working correctly
"""

import json
import os
import sys
import torch
import time

def test_configuration():
    """Test that the configuration is properly set for cluster processing"""
    print("🧪 Testing DEA Configuration Optimizations")
    print("=" * 50)
    
    # Load configuration
    config_path = "dea_config.json"
    if not os.path.exists(config_path):
        print("❌ Configuration file not found!")
        return False
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    global_config = config.get('GLOBAL_CONFIG', {})
    
    # Test Workers configuration
    workers = global_config.get('Workers', 0)
    print(f"Workers: {workers}")
    if workers >= 18:
        print("✅ Workers properly configured for cluster")
    else:
        print("⚠️  Workers may be suboptimal for 19-CPU cluster")
    
    # Test GPU configuration
    use_gpu = global_config.get('UseGPU', False)
    print(f"UseGPU: {use_gpu}")
    if use_gpu:
        print("✅ GPU enabled")
    else:
        print("⚠️  GPU disabled")
    
    return True

def test_torch_optimizations():
    """Test PyTorch optimizations"""
    print("\n🔥 Testing PyTorch Optimizations")
    print("=" * 50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Devices: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        device = torch.cuda.get_device_properties(0)
        print(f"Device Name: {device.name}")
        print(f"Device Memory: {device.total_memory // 1024 // 1024} MB")
    
    # Test thread configuration
    print(f"PyTorch Threads: {torch.get_num_threads()}")
    
    return cuda_available

def test_parallel_utilities():
    """Test parallel utility functions"""
    print("\n⚡ Testing Parallel Utilities")
    print("=" * 50)
    
    try:
        from utils_parallel import (
            log_resource_usage, 
            optimize_torch_performance, 
            PerformanceMonitor,
            OPTIMIZATION_CONFIG
        )
        
        print("✅ Parallel utilities imported successfully")
        
        # Test resource monitoring
        print("\n📊 Current Resource Usage:")
        log_resource_usage(use_gpu=torch.cuda.is_available())
        
        # Test performance monitor
        with PerformanceMonitor("Test Operation", use_gpu=torch.cuda.is_available()):
            time.sleep(1)  # Simulate some work
        
        print("\n⚙️  Optimization Configurations Available:")
        for config_name, config_vals in OPTIMIZATION_CONFIG.items():
            print(f"  {config_name}: {config_vals}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import parallel utilities: {e}")
        return False

def test_data_loading():
    """Test if data loading paths exist"""
    print("\n📁 Testing Data Paths")
    print("=" * 50)
    
    data_dir = "./data"
    required_dirs = [
        "data/datasets",
        "data/available_to_eve", 
        "data/dev"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"⚠️  {dir_path} not found")
            all_exist = False
    
    return all_exist

def main():
    """Run all optimization tests"""
    print("🚀 DEA Performance Optimization Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("PyTorch", test_torch_optimizations),
        ("Parallel Utilities", test_parallel_utilities),
        ("Data Paths", test_data_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All optimizations are working correctly!")
        print("\n📝 Next Steps:")
        print("1. Run 'python main.py' to execute the optimized pipeline")
        print("2. Monitor resource usage during execution") 
        print("3. Compare performance with previous runs")
    else:
        print("⚠️  Some optimizations may not be working properly")
        print("Check the failed tests above for details")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)