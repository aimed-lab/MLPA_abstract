#!/usr/bin/env python3
"""
BATCH FEATURE EXTRACTION - ALL PATIENTS
========================================
Extract 1,706 features for all NSCLC patients

This script:
- Processes all 422 patients in the NSCLC-Radiomics dataset
- Extracts 1,706 radiomics features per patient
- Saves results incrementally (safe against crashes)
- Provides detailed progress tracking
- Generates comprehensive summary

Output:
- radiomic_features_1706.npy: Feature matrix (422 × 1,706)
- feature_names_1706.json: List of 1,706 feature names
- patient_ids.json: List of 422 patient IDs
- extraction_log.txt: Detailed extraction log
- extraction_summary.json: Statistics and timing

Author: Claude AI
Date: 2025-12-16
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# Import extraction functions
from ferretti_full_extraction import extract_features_for_patient, analyze_feature_groups


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = './nsclc/manifest-1603198545583/NSCLC-Radiomics'
OUTPUT_DIR = './data3d_1706features/outputs'
N_JOBS = 9  # Number of CPU cores for parallel extraction

# Expected feature count (Ferretti full configuration)
EXPECTED_FEATURES = 1706


# =============================================================================
# PATIENT SCANNING
# =============================================================================

def scan_patients(data_root: str) -> List[Dict]:
    """
    Scan for patients with CT and segmentation.
    Returns ONE study per patient (study with most CT slices).
    """
    data_root_path = Path(data_root)
    patients_dict = {}
    
    print("\n📂 Scanning for patients...")
    
    for patient_dir in sorted(data_root_path.glob("LUNG1-*")):
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        
        for study_dir in sorted(patient_dir.glob("*")):
            if not study_dir.is_dir():
                continue
            
            all_subdirs = [d for d in study_dir.iterdir() if d.is_dir()]
            
            # Find SEG (series starting with "300.")
            seg_series_dirs = [d for d in all_subdirs if d.name.startswith("300.")]
            
            # Find CT (series NOT starting with "300." with at least 1 DICOM)
            valid_ct_dirs = []
            for d in all_subdirs:
                if d.name.startswith("300."):
                    continue
                dcm_files = list(d.rglob("*.dcm"))
                if len(dcm_files) >= 1:
                    valid_ct_dirs.append((d, len(dcm_files)))
            
            if valid_ct_dirs and seg_series_dirs:
                # Select CT directory with most slices
                ct_dir_info = max(valid_ct_dirs, key=lambda x: x[1])
                ct_dir = ct_dir_info[0]
                ct_file_count = ct_dir_info[1]
                
                seg_dir = seg_series_dirs[0]
                seg_files = list(seg_dir.glob("*.dcm"))
                
                if seg_files:
                    patient_data = {
                        'patient_id': patient_id,
                        'ct_dir': str(ct_dir),
                        'ct_file_count': ct_file_count,
                        'seg_file': str(seg_files[0])
                    }
                    
                    # Keep study with most CT slices
                    if patient_id not in patients_dict or \
                       ct_file_count > patients_dict[patient_id]['ct_file_count']:
                        patients_dict[patient_id] = patient_data
    
    patients_list = list(patients_dict.values())
    print(f"   ✅ Found {len(patients_list)} patients")
    
    return patients_list


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class ExtractionLogger:
    """Logger for extraction progress and errors."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.start_time = time.time()
        
        # Initialize log file
        with open(log_file, 'w') as f:
            f.write(f"="*80 + "\n")
            f.write(f"FERRETTI FULL FEATURE EXTRACTION LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"="*80 + "\n\n")
    
    def log(self, message: str, console: bool = True):
        """Log message to file and optionally console."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
        
        if console:
            print(log_message)
    
    def log_patient_success(self, patient_id: str, n_features: int, elapsed: float):
        """Log successful extraction."""
        message = f"✅ {patient_id}: {n_features} features in {elapsed:.1f}s"
        self.log(message)
    
    def log_patient_error(self, patient_id: str, error: str):
        """Log extraction error."""
        message = f"❌ {patient_id}: ERROR - {error}"
        self.log(message)
    
    def log_summary(self, successful: int, failed: int, total_time: float):
        """Log final summary."""
        self.log("\n" + "="*80)
        self.log("EXTRACTION COMPLETE")
        self.log("="*80)
        self.log(f"Total time: {total_time/3600:.2f} hours")
        self.log(f"Successful: {successful}/{successful+failed}")
        self.log(f"Failed: {failed}/{successful+failed}")
        self.log("="*80)


# =============================================================================
# INCREMENTAL SAVING
# =============================================================================

class IncrementalSaver:
    """Save features incrementally to prevent data loss."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_matrix = []
        self.patient_ids = []
        self.feature_names = None
        
    def add_patient(self, patient_id: str, features: Dict):
        """Add patient features."""
        # Store feature names from first patient
        if self.feature_names is None:
            self.feature_names = sorted(features.keys())
        
        # Convert features to array (sorted by feature name)
        feature_vector = np.array([features[name] for name in self.feature_names])
        
        self.feature_matrix.append(feature_vector)
        self.patient_ids.append(patient_id)
    
    def save_checkpoint(self, checkpoint_num: int):
        """Save checkpoint (every N patients)."""
        if len(self.feature_matrix) == 0:
            return
        
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save features
        features_array = np.array(self.feature_matrix)
        np.save(checkpoint_dir / f"features_checkpoint_{checkpoint_num}.npy", features_array)
        
        # Save patient IDs
        with open(checkpoint_dir / f"patient_ids_checkpoint_{checkpoint_num}.json", 'w') as f:
            json.dump(self.patient_ids, f, indent=2)
        
        print(f"   💾 Checkpoint {checkpoint_num} saved ({len(self.patient_ids)} patients)")
    
    def save_final(self):
        """Save final results."""
        if len(self.feature_matrix) == 0:
            raise ValueError("No features to save!")
        
        print("\n" + "="*80)
        print("💾 SAVING FINAL RESULTS")
        print("="*80)
        
        # Convert to numpy array
        features_array = np.array(self.feature_matrix)
        
        print(f"\n   Final shape: {features_array.shape}")
        print(f"   Patients: {len(self.patient_ids)}")
        print(f"   Features: {len(self.feature_names)}")
        
        # Save feature matrix
        features_file = self.output_dir / "radiomic_features_1706.npy"
        np.save(features_file, features_array)
        print(f"   ✅ Saved: {features_file}")
        
        # Save feature names
        names_file = self.output_dir / "feature_names_1706.json"
        with open(names_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"   ✅ Saved: {names_file}")
        
        # Save patient IDs
        ids_file = self.output_dir / "patient_ids.json"
        with open(ids_file, 'w') as f:
            json.dump(self.patient_ids, f, indent=2)
        print(f"   ✅ Saved: {ids_file}")
        
        # Save metadata
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'n_patients': len(self.patient_ids),
            'n_features': len(self.feature_names),
            'feature_extraction_method': 'Ferretti et al. (2024) full replication',
            'image_types': {
                'Original': 1,
                'Wavelet': 8,
                'LoG': 5,
                'Transformations': 4
            },
            'total_image_types': 18,
            'expected_features': EXPECTED_FEATURES,
            'actual_features': len(self.feature_names)
        }
        
        metadata_file = self.output_dir / "extraction_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Saved: {metadata_file}")
        
        print("\n" + "="*80)


# =============================================================================
# MAIN BATCH EXTRACTION
# =============================================================================

def run_batch_extraction():
    """Run batch extraction for all patients."""
    
    print("="*80)
    print("🚀 BATCH FEATURE EXTRACTION - FERRETTI FULL (1,706 features)")
    print("="*80)
    print(f"\n⚙️  Configuration:")
    print(f"   Data root: {DATA_ROOT}")
    print(f"   Output dir: {OUTPUT_DIR}")
    print(f"   CPU cores: {N_JOBS}")
    print(f"   Expected features: {EXPECTED_FEATURES}")
    
    # Initialize logger
    log_file = Path(OUTPUT_DIR) / "extraction_log.txt"
    logger = ExtractionLogger(str(log_file))
    logger.log(f"Configuration: {N_JOBS} CPU cores, {EXPECTED_FEATURES} expected features")
    
    # Initialize saver
    saver = IncrementalSaver(OUTPUT_DIR)
    
    # Scan patients
    patients_list = scan_patients(DATA_ROOT)
    
    if len(patients_list) == 0:
        print("\n❌ No patients found!")
        return
    
    logger.log(f"Found {len(patients_list)} patients to process")
    
    # Start extraction
    print(f"\n{'='*80}")
    print(f"📊 EXTRACTING FEATURES")
    print(f"{'='*80}\n")
    
    successful = 0
    failed = 0
    failed_patients = []
    
    start_time = time.time()
    
    for idx, patient_info in enumerate(patients_list, 1):
        patient_id = patient_info['patient_id']
        
        # Progress header
        print(f"\n[{idx}/{len(patients_list)}] {patient_id}")
        print(f"{'─'*80}")
        
        patient_start = time.time()
        
        try:
            # Extract features
            features = extract_features_for_patient(
                patient_info,
                use_parallel=True,
                n_jobs=N_JOBS
            )
            
            if features is None or len(features) == 0:
                raise ValueError("No features extracted")
            
            # Validate feature count
            if abs(len(features) - EXPECTED_FEATURES) > 50:
                logger.log(f"⚠️  {patient_id}: Feature count {len(features)} differs from expected {EXPECTED_FEATURES}")
            
            # Save features
            saver.add_patient(patient_id, features)
            
            patient_elapsed = time.time() - patient_start
            logger.log_patient_success(patient_id, len(features), patient_elapsed)
            
            successful += 1
            
            # Save checkpoint every 50 patients
            if successful % 50 == 0:
                saver.save_checkpoint(successful)
            
            # Time estimates
            elapsed_total = time.time() - start_time
            avg_time_per_patient = elapsed_total / idx
            remaining_patients = len(patients_list) - idx
            eta_seconds = remaining_patients * avg_time_per_patient
            eta = timedelta(seconds=int(eta_seconds))
            
            print(f"\n⏱️  Progress: {idx}/{len(patients_list)} ({100*idx/len(patients_list):.1f}%)")
            print(f"   Successful: {successful} | Failed: {failed}")
            print(f"   Avg time/patient: {avg_time_per_patient:.1f}s")
            print(f"   ETA: {eta}")
            
        except Exception as e:
            patient_elapsed = time.time() - patient_start
            error_msg = str(e)[:200]
            logger.log_patient_error(patient_id, error_msg)
            
            failed += 1
            failed_patients.append({
                'patient_id': patient_id,
                'error': error_msg,
                'time': patient_elapsed
            })
            
            print(f"\n❌ FAILED: {error_msg}")
    
    # Final save
    total_time = time.time() - start_time
    
    if successful > 0:
        saver.save_final()
    
    # Summary
    logger.log_summary(successful, failed, total_time)
    
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"\n⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"\n✅ Successful: {successful}/{len(patients_list)} ({100*successful/len(patients_list):.1f}%)")
    print(f"❌ Failed: {failed}/{len(patients_list)} ({100*failed/len(patients_list):.1f}%)")
    
    if failed_patients:
        print(f"\n❌ Failed patients:")
        for fp in failed_patients[:10]:  # Show first 10
            print(f"   • {fp['patient_id']}: {fp['error'][:80]}")
        if len(failed_patients) > 10:
            print(f"   ... and {len(failed_patients)-10} more (see log file)")
    
    # Save summary
    summary = {
        'extraction_date': datetime.now().isoformat(),
        'total_time_hours': total_time / 3600,
        'total_patients': len(patients_list),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(patients_list) if len(patients_list) > 0 else 0,
        'avg_time_per_patient': total_time / len(patients_list) if len(patients_list) > 0 else 0,
        'failed_patients': failed_patients,
        'output_files': {
            'features': 'radiomic_features_1706.npy',
            'feature_names': 'feature_names_1706.json',
            'patient_ids': 'patient_ids.json',
            'log': 'extraction_log.txt'
        }
    }
    
    summary_file = Path(OUTPUT_DIR) / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Summary saved: {summary_file}")
    print(f"📄 Log saved: {log_file}")
    print("\n" + "="*80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        run_batch_extraction()
        print("\n✅ EXTRACTION COMPLETE!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user")
        print("   Partial results may be saved in checkpoints/")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
