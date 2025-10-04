import re
import yaml
import subprocess
import pandas as pd

from pathlib import Path

# Get all experiment folders
train_dir = Path("runs/train")
experiment_folders = [f for f in train_dir.iterdir() if f.is_dir()]

# Quantization configurations to test
quant_configs = [
    {"half": False, "int8": False, "name": "FP32"},
    {"half": True, "int8": False, "name": "FP16"},
    {"half": False, "int8": True, "name": "INT8"}
]

# Store all results
all_results = []

for exp_folder in experiment_folders:
    # Find the best.pt model
    model_path = exp_folder / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        continue
    
    # Extract dataset name from folder name
    exp_name = exp_folder.name
    
    # Determine dataset based on experiment name
    if "BreastCancer" in exp_name:
        data_yaml = "Data/BreastCancer/data.yaml"
    elif "BloodCell" in exp_name:
        data_yaml = "Data/BloodCell/data.yaml"
    elif "Fracture" in exp_name:
        data_yaml = "Data/Fracture/data.yaml"
    else:
        print(f"⚠️  Unknown dataset for: {exp_name}")
        continue
    
    print(f"\n{'='*80}")
    print(f"Benchmarking: {exp_name}")
    print(f"{'='*80}")
    
    # Backup original data.yaml and modify to use test set
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Store original val path
    original_val = data_config.get('val')
    
    # Temporarily change val to test (if test exists, otherwise keep val)
    data_config['val'] = data_config.get('test', original_val)
    
    # Write modified config
    with open(data_yaml, 'w') as f:
        yaml.dump(data_config, f)
    
    try:
        for config in quant_configs:
            print(f"\n🔧 Testing configuration: {config['name']}")
            
            # Build benchmark command
            cmd = [
                "yolo", "benchmark",
                f"model={model_path}",
                f"data={data_yaml}",
                "imgsz=640",
                f"half={config['half']}",
                f"int8={config['int8']}",
                "device=cpu"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Combine stdout and stderr for parsing
                output = result.stdout + result.stderr
                
                if result.returncode != 0:
                    print(f"❌ Command failed with return code {result.returncode}")
                    print(f"Error output: {result.stderr[:500]}...")  # First 500 chars of stderr
                    continue  # Skip to next config
                
                # Extract metrics from validation output (before benchmark table)
                lines = output.split('\n')
                map50 = None
                map50_95 = None
                
                for line in lines:
                    if 'all' in line.lower() and any(char.isdigit() for char in line):  # More flexible check for 'all' line
                        parts = re.findall(r'\d+\.?\d*', line)  # Extract numbers
                        if len(parts) >= 6:  # Expect at least: images, instances, P, R, mAP50, mAP50-95
                            try:
                                map50 = float(parts[4])  # 5th number is mAP50
                                map50_95 = float(parts[5])  # 6th is mAP50-95
                                break
                            except (ValueError, IndexError):
                                continue
                
                # Extract from benchmark table (primary source for trained models)
                size_mb = None
                inference_time = None
                fps = None
                map50_95_table = None
                
                for line in lines:
                    if 'PyTorch' in line and '✅' in line and '|' in line:
                        # Use regex to extract numeric values: should give ['1', '5.2', '0.7344', '4.49', '222.62']
                        numbers = re.findall(r'\d+\.?\d*', line)
                        if len(numbers) >= 5:  # 1 + 4 metrics
                            try:
                                size_mb = float(numbers[1])  # Skip '1'
                                map50_95_table = float(numbers[2])
                                inference_time = float(numbers[3])
                                fps = float(numbers[4])
                                
                                # Use table mAP50-95 if not found earlier
                                if map50_95 is None:
                                    map50_95 = map50_95_table
                                break
                            except (ValueError, IndexError) as e:
                                print(f"❌ Error parsing table with regex: {e}")
                
                # Only add results if we have the key metrics
                if map50_95 is not None and size_mb is not None:
                    all_results.append({
                        'Experiment': exp_name,
                        'Dataset': exp_name.split('_')[0],
                        'Freeze': 'Frozen' if 'freeze_10' in exp_name else 'Unfrozen',
                        'Quantization': config['name'],
                        'Half_Precision': config['half'],
                        'INT8': config['int8'],
                        'Model_Size_MB': size_mb,
                        'mAP50': map50,
                        'mAP50-95': map50_95,
                        'Inference_Time_ms': inference_time,
                        'FPS': fps
                    })
                    print(f"✅ Success - mAP50: {map50 if map50 else 'N/A'}, mAP50-95: {map50_95:.4f}, FPS: {fps:.2f}")
                else:
                    print(f"❌ Failed to extract metrics for {config['name']}. Check debug output above.")
            
            except Exception as e:
                print(f"❌ Benchmark failed: {e}")
    
    finally:
        # Restore original data.yaml
        data_config['val'] = original_val
        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f)

# Create DataFrame and save results
if all_results:
    df = pd.DataFrame(all_results)
    
    # Sort by dataset and experiment
    df = df.sort_values(['Dataset', 'Freeze', 'Quantization'])
    
    # Save to CSV
    output_file = "runs/benchmark_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Benchmark complete! Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Display summary
    print("\n📊 Summary Statistics:")
    summary = df.groupby(['Dataset', 'Quantization'])[['mAP50', 'mAP50-95', 'FPS', 'Inference_Time_ms']].mean()
    print(summary)
    
    # Compare quantization impact
    print("\n📈 Quantization Impact (Average across all experiments):")
    quant_summary = df.groupby('Quantization')[['mAP50', 'mAP50-95', 'FPS', 'Inference_Time_ms', 'Model_Size_MB']].mean()
    print(quant_summary)
    
else:
    print("\n❌ No results collected. Check if models exist and benchmarks ran successfully.")
    print("💡 Possible issues: Missing dependencies (e.g., ONNX for INT8), invalid data paths, or GPU issues.")