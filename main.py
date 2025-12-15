"""
PFM-1 Mine Detection using Fast R-CNN
Optimized for Mac Silicon M3
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import time

# Check for MPS (Metal Performance Shaders) for M3
device = torch.device("cpu")
print(f"Using device: {device}")


class PFM1Dataset(Dataset):
    """Dataset class for PFM-1 mine detection from COCO format"""
    
    def __init__(self, coco_json_path, img_dir, transforms=None, filter_empty=True):
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Load COCO annotations
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.imgs = {img['id']: img for img in self.coco_data['images']}
        
        # Group annotations by image
        self.img_to_anns = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
        
        # Filter out images without annotations if requested
        if filter_empty:
            self.img_ids = [img_id for img_id in self.imgs.keys() 
                            if img_id in self.img_to_anns and len(self.img_to_anns[img_id]) > 0]
            empty_count = len(self.imgs) - len(self.img_ids)
            if empty_count > 0:
                print(f"Filtered out {empty_count} images without annotations")
        else:
            self.img_ids = list(self.imgs.keys())
        
        # Category mapping (PFM-1 mine will be class 1, background is 0)
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        print(f"Loaded {len(self.img_ids)} images with {len(self.coco_data['annotations'])} annotations")
        print(f"Categories: {self.categories}")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        anns = self.img_to_anns[img_id]
        
        boxes = []
        labels = []
        areas = []
        
        for ann in anns:
            # COCO format: [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            
            # Skip invalid boxes (zero width or height)
            if w <= 0 or h <= 0:
                continue
                
            # Convert to [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # PFM-1 mine class
            areas.append(w * h)
        
        # Handle images with no valid annotations
        if len(boxes) == 0:
            # Create a dummy box (required for training)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        image_id = torch.tensor([img_id])
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)
        
        return img, target


def get_model(num_classes):
    """
    Load Faster R-CNN model with ResNet-50 backbone
    (Note: PyTorch doesn't have standalone Fast R-CNN, Faster R-CNN is the standard)
    """
    # Load pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
        
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch}], Step [{i+1}/{len(data_loader)}], Loss: {losses.item():.4f}")
    
    return running_loss / len(data_loader)


@torch.no_grad()
def evaluate(model, data_loader, device, iou_threshold=0.5):
    """
    Evaluate model performance
    Calculates precision, recall, and mAP
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        outputs = model(images)
        
        for i, output in enumerate(outputs):
            pred_boxes = output['boxes'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            gt_labels = targets[i]['labels'].cpu().numpy()
            
            all_predictions.append({
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels
            })
            
            all_ground_truths.append({
                'boxes': gt_boxes,
                'labels': gt_labels
            })
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_ground_truths, iou_threshold)
    return metrics


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x_min, y_min, x_max, y_max]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_metrics(predictions, ground_truths, iou_threshold=0.5, score_threshold=0.5):
    """Calculate precision, recall, and mAP"""
    tp, fp, fn = 0, 0, 0
    aps = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = gt['boxes']
        
        # Filter predictions by score threshold
        valid_idx = pred_scores >= score_threshold
        pred_boxes = pred_boxes[valid_idx]
        pred_scores = pred_scores[valid_idx]
        
        matched_gt = set()
        
        # Sort predictions by score (descending)
        if len(pred_boxes) > 0:
            sorted_idx = np.argsort(pred_scores)[::-1]
            pred_boxes = pred_boxes[sorted_idx]
            pred_scores = pred_scores[sorted_idx]
        
        # Match predictions to ground truth
        for pred_box in pred_boxes:
            matched = False
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(gt_idx)
                    matched = True
                    break
            if not matched:
                fp += 1
        
        # Unmatched ground truths are false negatives
        fn += len(gt_boxes) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def visualize_predictions(model, dataset, device, num_samples=5, score_threshold=0.5):
    """Visualize model predictions on sample images"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            img, target = dataset[idx]
            
            # Make prediction
            prediction = model([img.to(device)])[0]
            
            # Convert image to numpy for visualization
            img_np = img.permute(1, 2, 0).cpu().numpy()
            
            ax.imshow(img_np)
            
            # Plot ground truth boxes (green)
            gt_boxes = target['boxes'].cpu().numpy()
            for box in gt_boxes:
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth'
                )
                ax.add_patch(rect)
            
            # Plot predicted boxes (red) with score
            pred_boxes = prediction['boxes'].cpu().numpy()
            pred_scores = prediction['scores'].cpu().numpy()
            
            for box, score in zip(pred_boxes, pred_scores):
                if score >= score_threshold:
                    rect = patches.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
                    )
                    ax.add_patch(rect)
                    ax.text(box[0], box[1] - 5, f'{score:.2f}', 
                            color='red', fontsize=12, weight='bold')
            
            ax.axis('off')
            ax.set_title(f'Image {idx} - Green: GT, Red: Prediction')
    
    plt.tight_layout()
    plt.savefig('detection_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'detection_results.png'")
    plt.show()


def plot_training_curves(train_losses, val_metrics):
    """Plot training loss and validation metrics"""
    if len(train_losses) == 0 or len(val_metrics) == 0:
        print("Warning: No training data to plot")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Training Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0.5, len(train_losses) + 0.5)
    
    # Precision
    precisions = [m['precision'] for m in val_metrics]
    axes[0, 1].plot(epochs, precisions, 'g-', linewidth=2, marker='s')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Precision', fontsize=12)
    axes[0, 1].set_title('Validation Precision', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0.5, len(precisions) + 0.5)
    axes[0, 1].set_ylim(0, 1.0)
    
    # Recall
    recalls = [m['recall'] for m in val_metrics]
    axes[1, 0].plot(epochs, recalls, 'r-', linewidth=2, marker='^')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Recall', fontsize=12)
    axes[1, 0].set_title('Validation Recall', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0.5, len(recalls) + 0.5)
    axes[1, 0].set_ylim(0, 1.0)
    
    # F1 Score
    f1_scores = [m['f1'] for m in val_metrics]
    axes[1, 1].plot(epochs, f1_scores, 'm-', linewidth=2, marker='d')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('F1 Score', fontsize=12)
    axes[1, 1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0.5, len(f1_scores) + 0.5)
    axes[1, 1].set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved training curves to 'training_curves.png'")
    plt.show()


def simple_tracker(predictions_sequence, iou_threshold=0.3):
    """
    Simple object tracker using IoU-based association
    predictions_sequence: list of predictions for each frame
    """
    tracks = []
    next_track_id = 0
    
    for frame_idx, pred in enumerate(predictions_sequence):
        boxes = pred['boxes']
        scores = pred['scores']
        
        if frame_idx == 0:
            # Initialize tracks
            for box, score in zip(boxes, scores):
                tracks.append({
                    'id': next_track_id,
                    'boxes': [box],
                    'scores': [score],
                    'frames': [frame_idx]
                })
                next_track_id += 1
        else:
            # Match detections to existing tracks
            matched_tracks = set()
            
            for box, score in zip(boxes, scores):
                best_iou = 0
                best_track_idx = -1
                
                # Find best matching track
                for track_idx, track in enumerate(tracks):
                    if track_idx in matched_tracks:
                        continue
                    
                    last_box = track['boxes'][-1]
                    iou = calculate_iou(box, last_box)
                    
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_track_idx = track_idx
                
                if best_track_idx >= 0:
                    # Update existing track
                    tracks[best_track_idx]['boxes'].append(box)
                    tracks[best_track_idx]['scores'].append(score)
                    tracks[best_track_idx]['frames'].append(frame_idx)
                    matched_tracks.add(best_track_idx)
                else:
                    # Create new track
                    tracks.append({
                        'id': next_track_id,
                        'boxes': [box],
                        'scores': [score],
                        'frames': [frame_idx]
                    })
                    next_track_id += 1
    
    return tracks


def main():
    """Main training and evaluation pipeline"""
    
    # ============= CONFIGURATION =============
    COCO_JSON_PATH = "result.json" 
    IMG_DIR = "./images/"
    NUM_EPOCHS = 20
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8
    
    # ============= LOAD DATASET =============
    print("\n=== Loading Dataset ===")
    full_dataset = PFM1Dataset(COCO_JSON_PATH, IMG_DIR)
    
    # Split into train and validation
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=2 
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        collate_fn=collate_fn, num_workers=2
    )
    
    # ============= BUILD MODEL =============
    print("\n=== Building Model ===")
    num_classes = 2  # Background + PFM-1 mine
    model = get_model(num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # ============= TRAINING =============
    print("\n=== Training Model ===")
    train_losses = []
    val_metrics = []
    
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        train_losses.append(train_loss)
        
        # Validate
        metrics = evaluate(model, val_loader, device)
        val_metrics.append(metrics)
        
        # print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        lr_scheduler.step()
    
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time / 60:.2f} minutes")
    
    # ============= FINAL EVALUATION =============
    print("\n=== Final Evaluation ===")
    final_metrics = evaluate(model, val_loader, device, iou_threshold=0.5)
    
    print(f"\nFinal Results (IoU >= 0.5):")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"True Positives: {final_metrics['tp']}")
    print(f"False Positives: {final_metrics['fp']}")
    print(f"False Negatives: {final_metrics['fn']}")

    # Save evaluation to file
    with open(f"fasterrcnn_{LEARNING_RATE}_{NUM_EPOCHS}_{BATCH_SIZE}.txt", "w") as f: f.write(f"Model: Faster R-CNN ResNet50\nLR: {LEARNING_RATE}\nEpochs: {NUM_EPOCHS}\nBatch: {BATCH_SIZE}\nPrecision: {final_metrics['precision']:.4f}\nRecall: {final_metrics['recall']:.4f}\nF1: {final_metrics['f1']:.4f}\nTruePos: {final_metrics['tp']}\nFalsePos: {final_metrics['fp']}\nFalseNeg: {final_metrics['fn']}\nTraining Time: {training_time/60:.2f} min")
    
    # ============= VISUALIZATION =============
    print("\n=== Generating Visualizations ===")
    plot_training_curves(train_losses, val_metrics)
    visualize_predictions(model, full_dataset, device, num_samples=5, score_threshold=0.5)
    
    # ============= SAVE MODEL =============
    print("\n=== Saving Model ===")
    torch.save(model.state_dict(), 'pfm1_detector.pth')
    print("Model saved as 'pfm1_detector.pth'")
    
    return model, train_losses, val_metrics


if __name__ == "__main__":
    model, losses, metrics = main()