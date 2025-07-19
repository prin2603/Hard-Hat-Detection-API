import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os

# === SETTINGS ===
TRAINED_MODEL_PATH = "helmet_model_epoch10.pth"
CLASSES = ["helmet"]
TEST_IMAGE_PATH = "hard_hat_workers9.png"

# === LOAD MODEL ===
def get_model(num_classes, model_weights=None, device="cpu"):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()
    return model

# === INFERENCE AND VISUALIZATION ===
def predict_and_show(model, image_path, classes, device="cpu", score_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)[0]

    # Show image with boxes
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    ax = plt.gca()
    num_found = 0
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        if score > score_threshold:
            num_found += 1
            x1, y1, x2, y2 = box.cpu().numpy()
            class_name = classes[label-1] if label > 0 and (label-1)<len(classes) else "background"
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(
                x1, y1-10, f"{class_name}: {score:.2f}",
                fontsize=12, color="white",
                bbox=dict(facecolor="red", alpha=0.5))
    if not num_found:
        print("No objects detected with confidence above threshold.")
    plt.axis("off")
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASSES) + 1   # Include background as class 0

    # Load model
    model = get_model(num_classes, model_weights=TRAINED_MODEL_PATH, device=device)

    # Test on a new image
    predict_and_show(model, TEST_IMAGE_PATH, CLASSES, device=device, score_threshold=0.5)
