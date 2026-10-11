# typing
from typing import List, Dict

import numpy as np
# inference
from inference.models import YOLOWorld

# cv2
import cv2

# supervision
import supervision as sv

# Fixed obstacle classes (same as _SEMANTIC_SAFETY_RADIUS_CELLS)
_OBSTACLE_CLASSES: List[str] = ["chair", "potted plant", "toilet"]


class YOLOWorldDetector:
    def __init__(self,
                 confidence_threshold: float
                 ):
        self.model = YOLOWorld(model_id="yolo_world/l")
        self.confidence_threshold = confidence_threshold
        self.classes = None
        self._last_obstacle_detections: dict = {}
        self._last_obstacle_confs: dict = {}

    def set_classes(self,
                    classes: List[str]
                    ):
        self.classes = classes
        # Combine query + obstacle classes; deduplicate while preserving order
        combined = list(classes)
        for c in _OBSTACLE_CLASSES:
            if c not in combined:
                combined.append(c)
        self._combined_classes = combined
        self.model.set_classes(combined)

    def detect(self, image: np.ndarray) -> dict:
        if self.classes is None:
            raise ValueError("Classes must be set before detecting")

        results = self.model.infer(image, confidence=self.confidence_threshold)

        preds = {
            "boxes": [],
            "scores": []
        }
        obstacle_detections: dict = {}
        obstacle_confs: dict = {}

        for detection in results.predictions:
            class_name = detection.class_name
            conf = float(detection.confidence)
            if conf <= self.confidence_threshold:
                continue

            x1 = detection.x - detection.width / 2
            y1 = detection.y - detection.height / 2
            x2 = detection.x + detection.width / 2
            y2 = detection.y + detection.height / 2
            if x1 == x2 or y1 == y2:
                continue
            box_list = [x1, y1, x2, y2]

            # Target detection (query class)
            if class_name == self.classes[0]:
                preds["boxes"].append(box_list)
                preds["scores"].append(conf)

            # Obstacle detection (all obstacle classes)
            if class_name in _OBSTACLE_CLASSES:
                if class_name not in obstacle_detections:
                    obstacle_detections[class_name] = []
                    obstacle_confs[class_name] = []
                obstacle_detections[class_name].append(box_list)
                obstacle_confs[class_name].append(conf)

        self._last_obstacle_detections = obstacle_detections
        self._last_obstacle_confs = obstacle_confs
        return preds

    def get_obstacle_detections(self) -> dict:
        """Return cached obstacle detections from the last detect() call.

        Returns:
            dict {class_name: [[x1, y1, x2, y2], ...]}
        """
        return self._last_obstacle_detections

    def get_obstacle_detections_with_conf(self) -> dict:
        """Return cached obstacle detections with confidence scores.

        Returns:
            dict {class_name: [(box, conf), ...]}
        """
        result = {}
        for class_name, boxes in self._last_obstacle_detections.items():
            confs = self._last_obstacle_confs.get(class_name, [0.0] * len(boxes))
            result[class_name] = list(zip(boxes, confs))
        return result

if __name__ == "__main__":
    # Test the YOLO World Detector
    detector = YOLOWorldDetector(confidence_threshold=0.5)
    detector.set_classes(["person", "car", "truck", "bus", "bicycle", "motorbike", "traffic light", "stop sign"])

    # Load an image
    image = cv2.imread("test_images/a.jpg")

    # Detect objects in the image
    detections = detector.detect(image)

    # Display the image with the detections
    image_with_detections = detections.draw_on_image(image)
    cv2.imshow("Detections", image_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
