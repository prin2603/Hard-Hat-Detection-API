import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

# === PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Images")        # Folder containing .png images
ANNOTATION_DIR = os.path.join(BASE_DIR, "Annotations")  # Folder containing .xml files

# --- Check that directories exist
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Image directory not found: {DATA_DIR}")
if not os.path.isdir(ANNOTATION_DIR):
    raise FileNotFoundError(f"Annotation directory not found: {ANNOTATION_DIR}")

# === CLASS LABELS ===
# Replace with your actual classes (as labelled in <name> tag in XMLs)
CLASSES = ["helmet"]  # Single class; add more if your dataset has them

def get_transform(train=True):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class VOCDataset(Dataset):
    def __init__(self, img_dir, anno_dir, transforms=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.transforms = transforms

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Generate .xml path by removing extension from img_name
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(self.anno_dir, xml_name)
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Annotation file not found: {xml_path}")

        # Parse boxes, labels
        boxes, labels = self.parse_voc_xml(xml_path)
        if len(boxes) == 0:
            # No object: Create dummy background (FasterRCNN expects at least 1 box, workaround: skip image)
            # Alternatively, you can remove such images from your dataset.
            boxes = [[0,0,1,1]]
            labels = [0]  # background
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }
        if self.transforms:
            image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.imgs)

    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in CLASSES:
                continue  # skip unknown classes
            # Pascal VOC is 1-based
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            # +1 because 0 is background
            labels.append(CLASSES.index(label) + 1)
        return boxes, labels

def collate_fn(batch):
    return tuple(zip(*batch))

# === DATASET / DATALOADER ===
dataset = VOCDataset(DATA_DIR, ANNOTATION_DIR, get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# === MODEL SETUP ===
num_classes = len(CLASSES) + 1  # +1 for background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === TRAINING ===
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
                           lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    lr_scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/len(data_loader):.4f}")
    torch.save(model.state_dict(), f"helmet_model_epoch{epoch+1}.pth")

print("Training complete.")
