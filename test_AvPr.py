import argparse
import numpy as np


class AveragePrecisionCalculator:
	def __init__(self, iou_threshold=0.5):
		self.iou_threshold = iou_threshold  # Порог IoU
		self.groundtruths = {}
		self.detections = {}

	def load_groundtruths(self, file_path):
		"""Загрузить истинную разметку."""
		self.groundtruths = self._parse_annotations(file_path)

	def load_detections(self, file_path):
		"""Загрузить предсказанную разметку детектора."""
		self.detections = self._parse_detections(file_path)

	def calculate_average_precision(self):
		"""Вычислить среднюю точность (AP)."""
		all_detections = self._sort_detections_by_confidence()
		total_gt = sum(len(self.groundtruths.get(frame, [])) for frame in self.groundtruths)
		tp, fp, all_tp, all_fp = 0, 0, [], []

		for frame_id, dets in all_detections.items():
			gts = self.groundtruths.get(frame_id, [])
			tps, fps, _ = self._match_detections_to_groundtruth(dets, gts)
			tp += tps
			fp += fps
			all_tp.append(tp)
			all_fp.append(fp)

		precisions, recalls = self._calculate_precision_recall(all_tp, all_fp, total_gt)
		return self._compute_ap(precisions, recalls)

	# ======= Приватные методы =======
	def _parse_annotations(self, file_path):
		annotations = {}
		with open(file_path, 'r') as f:
			for line in f:
				frame_id, class_name, *bbox = line.strip().split()
				frame_id = int(frame_id)
				bbox = list(map(float, bbox))
				if frame_id not in annotations:
					annotations[frame_id] = []
				annotations[frame_id].append(bbox)
		return annotations

	def _parse_detections(self, file_path):
		detections = {}
		with open(file_path, 'r') as f:
			for line in f:
				frame_id, class_name, conf, *bbox = line.strip().split()
				frame_id = int(frame_id)
				conf = float(conf)
				bbox = list(map(float, bbox))
				if frame_id not in detections:
					detections[frame_id] = []
				detections[frame_id].append([class_name, conf] + bbox)
		return detections

	def _calculate_iou(self, bbox1, bbox2):
		x1, y1, x2, y2 = bbox1
		x1g, y1g, x2g, y2g = bbox2

		xi1 = max(x1, x1g)
		yi1 = max(y1, y1g)
		xi2 = min(x2, x2g)
		yi2 = min(y2, y2g)

		inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
		bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
		bbox2_area = (x2g - x1g + 1) * (y2g - y1g + 1)
		union_area = bbox1_area + bbox2_area - inter_area

		return inter_area / union_area if union_area > 0 else 0

	def _match_detections_to_groundtruth(self, detections, groundtruths):
		matched = set()
		tp, fp, fn = 0, 0, 0

		for det in detections:
			_, _, conf, x1, y1, x2, y2 = det
			best_iou = 0
			best_gt_idx = -1
			for idx, gt in enumerate(groundtruths):
				iou = self._calculate_iou([x1, y1, x2, y2], gt)
				if iou > best_iou:
					best_iou = iou
					best_gt_idx = idx
			if best_iou >= self.iou_threshold and best_gt_idx not in matched:
				tp += 1
				matched.add(best_gt_idx)
			else:
				fp += 1

		fn = len(groundtruths) - len(matched)
		return tp, fp, fn

	def _sort_detections_by_confidence(self):
		sorted_detections = {}
		for frame, dets in self.detections.items():
			sorted_detections[frame] = sorted(dets, key=lambda x: -x[1])  # Sort by confidence
		return sorted_detections

	def _calculate_precision_recall(self, all_tp, all_fp, total_gt):
		precisions = []
		recalls = []
		for tp, fp in zip(all_tp, all_fp):
			precision = tp / (tp + fp) if tp + fp > 0 else 0
			recall = tp / total_gt if total_gt > 0 else 0
			precisions.append(precision)
			recalls.append(recall)
		return precisions, recalls

	def _compute_ap(self, precisions, recalls):
		precisions = [0] + precisions + [0]
		recalls = [0] + recalls + [1]

		for i in range(len(precisions) - 2, -1, -1):
			precisions[i] = max(precisions[i], precisions[i + 1])

		ap = 0
		for i in range(1, len(recalls)):
			ap += (recalls[i] - recalls[i - 1]) * precisions[i]
		return ap


# ======= Основной блок для консоли =======
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calculate Average Precision (AP) for object detection.")
	parser.add_argument("groundtruth", type=str, help="Path to the ground truth file.")
	parser.add_argument("detections", type=str, help="Path to the detections file.")
	parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5).")

	args = parser.parse_args()

	ap_calculator = AveragePrecisionCalculator(iou_threshold=args.iou)
	ap_calculator.load_groundtruths(args.groundtruth)
	ap_calculator.load_detections(args.detections)

	ap = ap_calculator.calculate_average_precision()
	print(f"Average Precision (AP): {ap:.4f}")