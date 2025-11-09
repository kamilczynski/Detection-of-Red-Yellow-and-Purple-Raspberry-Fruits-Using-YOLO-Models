from ultralytics import YOLO
import torch
import os
import csv
import yaml
import json

# === CONFIGURATION ===
model_path = "C:/Users/HARDPC/runs/train_MIXCOLOR8S/yolov8s_MIXCOLOR8S/weights/best.pt"
data_yaml = "C:/Users/HARDPC/data_czerwone.yaml"
save_dir = "C:/Users/HARDPC/Desktop/PROJEKTY CNN/WYNIKI/MIXCOLOR/METRYKIRED/8S"

# === CUDA DETECTION ===
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise SystemError("❌ No GPU detected! Validation requires a CUDA device (e.g., RTX 5080 or Jetson Orin NX).")

print(f"\n🔥 Starting evaluation on TEST SET using device: {torch.cuda.get_device_name(0)}")

os.makedirs(save_dir, exist_ok=True)

# === 1️⃣ Load model ===
model = YOLO(model_path)

# === 2️⃣ Evaluate model on TEST SET ===
print("\n🚀 Starting model evaluation on TEST SET...")
results = model.val(
    data=data_yaml,
    split="test",
    imgsz=640,
    device=device,
    save_json=True,
    save_hybrid=False,
    verbose=True,
    plots=True
)

# === 3️⃣ Collect and print global metrics ===
precision = results.box.p.mean()
recall = results.box.r.mean()
f1 = results.box.f1.mean()
map50 = results.box.map50
map5095 = results.box.map

print("\n=== GLOBAL METRICS (averaged over all classes) ===")
print(f"Precision:    {precision:.3f}")
print(f"Recall:       {recall:.3f}")
print(f"F1-score:     {f1:.3f}")
print(f"mAP@0.5:      {map50:.3f}")
print(f"mAP@[.5:.95]: {map5095:.3f}")

# === 3️⃣b PER-CLASS METRICS ===
names = model.names  # class names (e.g., {0: 'red', 1: 'yellow', 2: 'purple'})

print("\n=== PER-CLASS METRICS ===")
per_class_metrics = []

for i, name in names.items():
    precision_i = results.box.p[i]
    recall_i = results.box.r[i]
    f1_i = results.box.f1[i]
    ap50 = results.box.ap50[i]
    ap5095 = results.box.ap[i]
    per_class_metrics.append([name, precision_i, recall_i, f1_i, ap50, ap5095])
    print(f"{name:10s} | P: {precision_i:.3f} | R: {recall_i:.3f} | F1: {f1_i:.3f} | mAP@0.5: {ap50:.3f} | mAP@0.5:0.95: {ap5095:.3f}")

# Save global metrics to CSV
csv_path = os.path.join(save_dir, "metrics_summary_testset.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    for name, val in [
        ("Precision", precision),
        ("Recall", recall),
        ("F1-score", f1),
        ("mAP@0.5", map50),
        ("mAP@[.5:.95]", map5095)
    ]:
        writer.writerow([name, f"{val:.4f}"])
print(f"\n📁 Global metrics saved to: {csv_path}")

# Save per-class metrics to CSV
csv_path_perclass = os.path.join(save_dir, "metrics_per_class.csv")
with open(csv_path_perclass, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Precision", "Recall", "F1", "mAP@0.5", "mAP@[.5:.95]"])
    for row in per_class_metrics:
        writer.writerow([row[0]] + [f"{x:.4f}" for x in row[1:]])
print(f"📁 Per-class metrics saved to: {csv_path_perclass}")

# === 5️⃣ Load TEST SET path from data.yaml ===
with open(data_yaml, 'r') as f:
    data_config = yaml.safe_load(f)
base_path = data_config.get('path', '')
test_split = data_config.get('test', None)
if not test_split:
    raise FileNotFoundError("❌ 'test:' section not defined in data.yaml file.")

if os.path.isabs(test_split):
    test_dir = test_split
else:
    test_dir = os.path.join(base_path, test_split)
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"❌ TEST SET folder not found: {test_dir}")
print(f"\n📂 TEST SET detected: {test_dir}")

# === 6️⃣ Run detection and save TEST SET images + JSON results ===
print("\n📸 Running detection on TEST SET...")
pred_results = model.predict(
    source=test_dir,
    imgsz=640,
    conf=0.25,
    device=device,
    save=True,
    save_txt=False,
    project=save_dir,
    name="predicted_testset",
    exist_ok=True,
    verbose=True
)

# === 7️⃣ Save all predictions to JSON ===
json_path = os.path.join(save_dir, "predictions_testset.json")
json_data = []
for result in pred_results:
    boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes else []
    confs = result.boxes.conf.cpu().numpy().tolist() if result.boxes else []
    classes = result.boxes.cls.cpu().numpy().tolist() if result.boxes else []
    json_data.append({
        "image": result.path,
        "boxes": boxes,
        "confidences": confs,
        "classes": classes
    })

with open(json_path, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"\n💾 Full TEST SET detection results saved to: {json_path}")
print(f"✅ TEST SET images with bounding boxes saved in: {os.path.join(save_dir, 'predicted_testset')}")
print("\n=== TEST SET evaluation completed successfully ===")
