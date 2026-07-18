"""
PCB Sentinel - Detection Engine
Adapted from the original PyQt desktop GUI's YOLOv8Detector class.
Core detection logic is unchanged - only the PyQt/threading wrapper was
removed since Flask handles requests synchronously per worker.
"""
import time
import json
import os
import numpy as np
from ultralytics import YOLO


class YOLOv8Detector:
    def __init__(self, model_path: str, config_path: str = None):
        self.model = YOLO(model_path)
        self.conf_thresh = 0.25
        self.class_names = None
        self.device = "cuda" if self.model.device.type == "cuda" else "cpu"
        self.inference_times = []
        self.total_predictions = 0

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.class_names = config.get('class_names', [])
                self.conf_thresh = config.get('confidence_threshold', 0.25)
        else:
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())

        print(f"[INFO] Model classes: {self.class_names}")
        print(f"[INFO] Confidence threshold: {self.conf_thresh}")

    def predict(self, image: np.ndarray) -> dict:
        start = time.time()
        results = self.model(image, conf=self.conf_thresh, verbose=False)
        inference_time = (time.time() - start) * 1000

        boxes = results[0].boxes
        defects = []
        bboxes = []
        confidences = []

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = (self.class_names[class_id]
                          if self.class_names and class_id < len(self.class_names)
                          else f"class_{class_id}")
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            defects.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': xyxy,
                'severity': self._get_severity(class_name, confidence)
            })
            bboxes.append(xyxy)
            confidences.append(confidence)

        self.inference_times.append(inference_time)
        self.total_predictions += 1

        return {
            'defects': defects,
            'defect_count': len(defects),
            'defect_detected': len(defects) > 0,
            'confidence_scores': confidences,
            'bboxes': bboxes,
            'processing_time': inference_time,
            'avg_processing_time': np.mean(self.inference_times[-100:]) if self.inference_times else 0
        }

    def _get_severity(self, class_name: str, confidence: float) -> str:
        critical = ["short_circuit", "open_trace", "missing_cap", "missing_comp",
                    "missing_ic", "missing_diode", "missing_ferrite", "missing_inductor"]
        if class_name in critical:
            return "high" if confidence > 0.7 else "medium"
        return "medium" if confidence > 0.7 else "low"

    def get_model_info(self) -> dict:
        return {
            'name': 'YOLOv8 PCB Detector',
            'version': '1.0',
            'classes': self.class_names,
            'input_size': (640, 640),
            'avg_inference_time': np.mean(self.inference_times[-100:]) if self.inference_times else 0,
            'total_predictions': self.total_predictions
        }


def draw_detections(img, results):
    """Draws bounding boxes + labels on an image (OpenCV BGR array), matching
    the original GUI's color-by-severity scheme. Returns the modified image."""
    import cv2
    img_h, img_w = img.shape[:2]
    font_scale = max(0.4, min(1.2, img_w / 1000.0))
    thickness = max(1, int(font_scale * 2))
    box_thickness = max(2, int(font_scale * 3))
    used_label_positions = []

    for bbox, defect in zip(results['bboxes'], results['defects']):
        x1, y1, x2, y2 = map(int, bbox)
        color = {'high': (30, 30, 220), 'medium': (0, 140, 255), 'low': (80, 200, 0)}.get(
            defect.get('severity', 'medium'), (0, 140, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)
        label = f"{defect['class']}: {defect['confidence']*100:.1f}%"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = y1 - lh - 6 if y1 - lh - 6 > 0 else y1 + lh + 6
        while any(abs(label_y - uy) < lh + 4 for uy in used_label_positions):
            label_y += lh + 4
        used_label_positions.append(label_y)
        label_top = label_y - lh - baseline
        label_bottom = label_y + baseline
        label_x = x1
        if label_x + lw > img_w:
            label_x = max(0, img_w - lw - 6)
        cv2.rectangle(img, (label_x, label_top - 2), (label_x + lw + 4, label_bottom + 2), color, cv2.FILLED)
        cv2.putText(img, label, (label_x + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img
