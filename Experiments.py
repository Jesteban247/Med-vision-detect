import os
import subprocess

# Datasets
datasets = ["Data/BreastCancer", "Data/BloodCell", "Data/Fracture"]

# Freeze options: 10 for frozen, 0 for unfrozen
freeze_options = [10, 0]

# Common parameters
epochs = 30
imgsz = 640
batch = 32
amp = True
plots = True
device = "0,1"
model = "yolo11n.pt"
task = "detect"
mode = "train"

for dataset_path in datasets:
    data_yaml = os.path.join(dataset_path, "data.yaml")
    dataset_name = os.path.basename(dataset_path)
    
    for freeze in freeze_options:
        # Build command
        cmd = [
            "yolo",
            f"task={task}",
            f"mode={mode}",
            f"model={model}",
            f"data={data_yaml}",
            f"epochs={epochs}",
            f"imgsz={imgsz}",
            f"batch={batch}",
            f"amp={amp}",
            f"plots={plots}",
            f"device={device}"
        ]
        
        if freeze > 0:
            cmd.append(f"freeze={freeze}")
        
        # Run name or project
        project_name = f"{dataset_name}_freeze_{freeze}" if freeze > 0 else f"{dataset_name}_unfrozen"
        cmd.append(f"project=runs/train")
        cmd.append(f"name={project_name}")
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)