#!/usr/bin/env python3
"""
QUICK TEST - Single Patient Extraction
=======================================
Test the full 1,706 feature extraction on ONE patient before batch processing

This validates:
✅ All image filters work correctly
✅ Feature count is ~1,706
✅ Extraction completes without errors
✅ Parallel processing works
✅ Feature naming is correct

Usage:
    python test_1706_extraction.py

Output:
    - Console output with detailed validation
    - test_patient_features.json (sample features)

Author: Claude AI
Date: 2025-12-16
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Import extraction functions
from ferretti_full_extraction import (
    extract_features_for_patient,
    analyze_feature_groups
)


# =============================================================================
# PATIENT SCANNER
# =============================================================================

def find_first_patient(data_root: str):
    """Find the first valid patient in the dataset."""
    data_root_path = Path(data_root)
    
    print("🔍 Scanning for first valid patient...")
    
    for patient_dir in sorted(data_root_path.glob("LUNG1-*")):
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        
        for study_dir in sorted(patient_dir.glob("*")):
            if not study_dir.is_dir():
                continue
            
            all_subdirs = [d for d in study_dir.iterdir() if d.is_dir()]
            
            # Find SEG
            seg_dirs = [d for d in all_subdirs if d.name.startswith("300.")]
            
            # Find CT
            valid_ct = []
            for d in all_subdirs:
                if d.name.startswith("300."):
                    continue
                dcm_files = list(d.rglob("*.dcm"))
                if len(dcm_files) >= 1:
                    valid_ct.append((d, len(dcm_files)))
            
            if valid_ct and seg_dirs:
                ct_dir, ct_count = max(valid_ct, key=lambda x: x[1])
                seg_files = list(seg_dirs[0].glob("*.dcm"))
                
                if seg_files:
                    return {
                        'patient_id': patient_id,
                        'ct_dir': str(ct_dir),
                        'ct_file_count': ct_count,
                        'seg_file': str(seg_files[0])
                    }
    
    return None


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def validate_feature_count(features: dict, expected: int = 1706, tolerance: int = 50):
    """Validate feature count is close to expected."""
    actual = len(features)
    difference = abs(actual - expected)
    
    print(f"\n📊 Feature Count Validation:")
    print(f"   Expected: ~{expected}")
    print(f"   Actual:   {actual}")
    print(f"   Difference: {difference}")
    
    if difference <= tolerance:
        print(f"   ✅ PASS: Within tolerance (±{tolerance})")
        return True
    else:
        print(f"   ⚠️  WARNING: Outside tolerance (±{tolerance})")
        return False


def validate_feature_categories(features: dict):
    """Validate feature distribution across categories."""
    categories = analyze_feature_groups(features)
    
    print(f"\n📊 Feature Category Validation:")
    print(f"   {'Category':<25} {'Count':>6}  {'Expected':>6}  Status")
    print(f"   {'-'*25} {'-'*6}  {'-'*6}  {'-'*6}")
    
    expected = {
        'Original_shape': 14,
        'Original_firstorder': 19,
        'Original_texture': 75,
        'Wavelet': 752,
        'LoG': 470,
        'Square': 94,
        'SquareRoot': 94,
        'Logarithm': 94,
        'Exponential': 94
    }
    
    all_pass = True
    for category, actual_count in categories.items():
        exp_count = expected.get(category, 0)
        if exp_count == 0:
            continue
        
        # Allow 10% tolerance
        tolerance = max(5, int(exp_count * 0.1))
        difference = abs(actual_count - exp_count)
        
        if difference <= tolerance:
            status = "✅ PASS"
        else:
            status = "⚠️  WARN"
            all_pass = False
        
        print(f"   {category:<25} {actual_count:>6}  {exp_count:>6}  {status}")
    
    print(f"   {'-'*25} {'-'*6}  {'-'*6}  {'-'*6}")
    print(f"   {'TOTAL':<25} {len(features):>6}  {sum(expected.values()):>6}")
    
    return all_pass


def validate_feature_naming(features: dict):
    """Validate feature naming conventions."""
    print(f"\n📊 Feature Naming Validation:")
    
    # Check for expected prefixes
    prefixes = {
        'original_': 0,
        'wavelet-': 0,
        'log-sigma-': 0,
        'square_': 0,
        'squareroot_': 0,
        'logarithm_': 0,
        'exponential_': 0
    }
    
    for feature_name in features.keys():
        name_lower = feature_name.lower()
        for prefix in prefixes.keys():
            if name_lower.startswith(prefix):
                prefixes[prefix] += 1
                break
    
    print(f"   Prefix Distribution:")
    for prefix, count in prefixes.items():
        print(f"      {prefix:<20} {count:>4} features")
    
    # Check for required feature classes
    required_classes = ['shape', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    found_classes = set()
    
    for feature_name in features.keys():
        name_lower = feature_name.lower()
        for cls in required_classes:
            if cls in name_lower:
                found_classes.add(cls)
    
    print(f"\n   Feature Classes Found:")
    for cls in required_classes:
        status = "✅" if cls in found_classes else "❌"
        print(f"      {status} {cls}")
    
    return len(found_classes) == len(required_classes)


def sample_features(features: dict, n_samples: int = 10):
    """Display sample features."""
    print(f"\n📊 Sample Features (first {n_samples}):")
    
    for i, (key, value) in enumerate(list(features.items())[:n_samples], 1):
        if isinstance(value, (int, float)):
            print(f"   {i:2d}. {key}: {value:.6f}")
        else:
            print(f"   {i:2d}. {key}: {value}")


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("="*80)
    print("🧪 QUICK TEST - Single Patient Extraction (1,706 features)")
    print("="*80)
    
    # Configuration
    DATA_ROOT = './nsclc/manifest-1603198545583/NSCLC-Radiomics'
    N_JOBS = 9
    
    print(f"\n⚙️  Configuration:")
    print(f"   Data root: {DATA_ROOT}")
    print(f"   CPU cores: {N_JOBS}")
    print(f"   Expected features: ~1,706")
    
    # Find first patient
    print(f"\n{'='*80}")
    patient_info = find_first_patient(DATA_ROOT)
    
    if patient_info is None:
        print("\n❌ No valid patient found!")
        print("   Check DATA_ROOT path:")
        print(f"   {DATA_ROOT}")
        return 1
    
    print(f"   ✅ Found: {patient_info['patient_id']}")
    print(f"   CT slices: {patient_info['ct_file_count']}")
    print(f"   CT dir: {patient_info['ct_dir']}")
    print(f"   SEG file: {patient_info['seg_file']}")
    
    # Extract features
    print(f"\n{'='*80}")
    print("🚀 EXTRACTING FEATURES")
    print("="*80)
    
    start_time = time.time()
    
    try:
        features = extract_features_for_patient(
            patient_info,
            use_parallel=True,
            n_jobs=N_JOBS
        )
        
        if features is None or len(features) == 0:
            print("\n❌ FAILED: No features extracted!")
            return 1
        
        elapsed = time.time() - start_time
        
        print(f"\n⏱️  Extraction time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # Run validations
        print(f"\n{'='*80}")
        print("✅ VALIDATION TESTS")
        print("="*80)
        
        test_results = []
        
        # Test 1: Feature count
        test_results.append(validate_feature_count(features))
        
        # Test 2: Feature categories
        test_results.append(validate_feature_categories(features))
        
        # Test 3: Feature naming
        test_results.append(validate_feature_naming(features))
        
        # Sample features
        sample_features(features, n_samples=15)
        
        # Overall result
        print(f"\n{'='*80}")
        print("🎯 TEST RESULTS")
        print("="*80)
        
        passed = sum(test_results)
        total = len(test_results)
        
        print(f"\n   Tests passed: {passed}/{total}")
        
        if passed == total:
            print(f"   ✅ ALL TESTS PASSED!")
            print(f"\n   You are ready to run batch extraction:")
            print(f"   python batch_extract_1706_features.py")
        else:
            print(f"   ⚠️  {total - passed} test(s) failed")
            print(f"   Review warnings above")
        
        # Save sample features
        output_file = "test_patient_features.json"
        sample_dict = {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                      for k, v in list(features.items())[:50]}
        
        with open(output_file, 'w') as f:
            json.dump(sample_dict, f, indent=2)
        
        print(f"\n💾 Sample features saved: {output_file}")
        
        print(f"\n{'='*80}")
        print("✅ TEST COMPLETE")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("❌ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
