import os


# The code was written based on the work by He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2020). "Mask R-CNN."
# https://www.baeldung.com/cs/object-detection-intersection-vs-union
class DetectionMetricsCounter:

    def parse_bounding_boxes(self, file_path):
        """Parse bounding box values from a file."""
        bounding_boxes = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 4:
                    bbox = tuple(map(int, parts[-4:]))
                    bounding_boxes.append(bbox)
        return bounding_boxes

    def calculate_iou(self, box1, box2):
        """Calculate IoU of two bounding boxes."""
        # Coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        # No overlap condition
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Overlap area
        overlap_area = (x_right - x_left) * (y_bottom - y_top)

        # Areas of individual boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Combined area
        total_area = box1_area + box2_area - overlap_area

        # Intersection over Union
        iou = overlap_area / total_area

        return iou

    def calculate_tp_fp_fn(self, ground_truths_path, predictions_path):

        true_positives = 0
        false_negatives = 0
        false_positives = 0
        ground_truth_boxes_total = 0

        # Iterate through ground truth files
        for gt_file in os.listdir(ground_truths_path):
            if gt_file.endswith(".txt"):
                gt_file_path = os.path.join(ground_truths_path, gt_file)
                pred_file_path = os.path.join(predictions_path, gt_file)

                gt_bboxes = self.parse_bounding_boxes(gt_file_path)
                pred_bboxes = self.parse_bounding_boxes(pred_file_path) if os.path.exists(pred_file_path) else []

                # Compare each ground truth bbox to prediction bboxes
                for gt_bbox in gt_bboxes:
                    ground_truth_boxes_total += 1
                    has_match = False
                    for pred_bbox in pred_bboxes:
                        if self.calculate_iou(gt_bbox, pred_bbox) >= 0.5:
                            true_positives += 1
                            has_match = True
                            break
                    if not has_match:
                        print("False negative in file: ", gt_file)
                        false_negatives += 1

                for pred_bbox in pred_bboxes:
                    has_match = False
                    for gt_bbox in gt_bboxes:
                        if self.calculate_iou(pred_bbox, gt_bbox) >= 0.5:
                            has_match = True
                            break
                    if not has_match:
                        print("False positive in file: ", gt_file)
                        false_positives += 1

        print("True Positives:", true_positives)
        print("False positive:", false_positives)
        print("False Negatives:", false_negatives)
        print("Ground truth boxes:", ground_truth_boxes_total)
