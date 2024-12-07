import argparse
import numpy as np
import argparse


def cli_arguments():
	"""
    Обрабатывает аргументы командной строки для скрипта вычисления средней точности (AP).

    :return: Объект с аргументами.
    """
	parser = argparse.ArgumentParser(description="Calculate Average Precision (AP) for object detection.")

	parser.add_argument("--groundtruth", "-g",
						help="Path to the ground truth file.",
						type=str,
						dest="groundtruth_path",
						required=True,
						default="DLMini/layout/track_09_0-2000.txt")

	parser.add_argument("--detections", "-d",
						help="Path to the detections file.",
						type=str,
						dest="detections_path",
						required=True,
						default="DLMini/layout/track_09_0-2000.txt")

	parser.add_argument("--iou", "-iou",
						help="Intersection over Union (IoU) threshold (default: 0.5).",
						type=float,
						dest="iou_threshold",
						required=False,
						default=0.5)

	parser.add_argument("--verbose", "-v",
						help="Enable verbose output.",
						action="store_true",
						dest="verbose")

	return parser.parse_args()


class AveragePrecisionCalculator:
	def __init__(self, iou_threshold=0.5):
		"""
		Инициализация класса для вычисления средней точности (AP).

		:param iou_threshold: Порог Intersection over Union (IoU), используется для определения,
							  соответствует ли найденный объект истинной разметке.
		"""
		self.iou_threshold = iou_threshold  # Порог IoU
		self.groundtruths = {}	# Dict: Истинная разметка (группируется по кадрам)
		self.detections = {}	# Dict: Разметка детектора (группируется по кадрам)

	def load_groundtruths(self, file_path):
		"""
		Загрузка истинной разметки из файла.

		:param file_path: Путь к файлу с истинной разметкой.
		"""
		self.groundtruths = self._parse_annotations(file_path)

	def load_detections(self, file_path):
		"""
		Загрузка предсказаний детектора из файла.

		:param file_path: Путь к файлу с предсказаниями.
		"""
		self.detections = self._parse_detections(file_path)

	def calculate_average_precision(self):
		"""
		Вычисляет среднюю точность (Average Precision, AP).

		:return: Значение средней точности AP.
		"""
		# Сортируем предсказания по уверенности (confidence)
		all_detections = self._sort_detections_by_confidence()
		# Общее количество объектов в истинной разметке
		total_gt = sum(len(self.groundtruths.get(frame, [])) for frame in self.groundtruths)
		tp, fp, all_tp, all_fp = 0, 0, [], []

		# Сравниваем предсказания с истинной разметкой по каждому кадру
		for frame_id, dets in all_detections.items():
			gts = self.groundtruths.get(frame_id, [])
			tps, fps, _ = self._match_detections_to_groundtruth(dets, gts)
			tp += tps
			fp += fps
			all_tp.append(tp)
			all_fp.append(fp)

		# Вычисляем precision и recall для всех точек
		precisions, recalls = self._calculate_precision_recall(all_tp, all_fp, total_gt)
		# Вычисляем площадь под кривой зависимости точности от отклика
		return self._compute_ap(precisions, recalls)

	# ======= Приватные методы =======
	def _parse_annotations(self, file_path):
		"""
		Парсинг файла с истинной разметкой.

		:param file_path: Путь к файлу с разметкой.
		:return: Словарь {frame_id: [список ограничивающих прямоугольников]}.
		"""
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
		"""
		Парсинг файла с предсказаниями детектора.

		:param file_path: Путь к файлу с предсказаниями.
		:return: Словарь {frame_id: [список предсказаний]}, где каждое предсказание содержит:
				 [class_name, confidence, x1, y1, x2, y2].
		"""
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
		"""
		Вычисляет Intersection over Union (IoU) для двух прямоугольников.

		:param bbox1: Первый ограничивающий прямоугольник [x1, y1, x2, y2].
		:param bbox2: Второй ограничивающий прямоугольник [x1, y1, x2, y2].
		:return: Значение IoU (от 0 до 1).
		"""
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
		"""
		Сопоставляет предсказания с истинной разметкой.

		:param detections: Список предсказаний для кадра.
		:param groundtruths: Список истинных объектов для кадра.
		:return: Количество TP (true positives), FP (false positives) и FN (false negatives).
		"""
		matched = set()
		tp, fp, fn = 0, 0, 0

		for det in detections:
			_, conf, x1, y1, x2, y2 = det
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
		"""
		Сортирует предсказания по уверенности (confidence).

		:return: Словарь {frame_id: [отсортированный список предсказаний]}.
		"""
		sorted_detections = {}
		for frame, dets in self.detections.items():
			sorted_detections[frame] = sorted(dets, key=lambda x: -x[1])
		return sorted_detections

	def _calculate_precision_recall(self, all_tp, all_fp, total_gt):
		"""
		Вычисляет массивы precision (точности) и recall (отклика).

		:param all_tp: Кумулятивное количество TP (true positives).
		:param all_fp: Кумулятивное количество FP (false positives).
		:param total_gt: Общее количество объектов в истинной разметке.
		:return: Списки значений precision и recall.
		"""
		precisions = []
		recalls = []
		for tp, fp in zip(all_tp, all_fp):
			precision = tp / (tp + fp) if tp + fp > 0 else 0
			recall = tp / total_gt if total_gt > 0 else 0
			precisions.append(precision)
			recalls.append(recall)
		return precisions, recalls

	def _compute_ap(self, precisions, recalls):
		"""
		Вычисляет площадь под кривой зависимости precision от recall.

		:param precisions: Список значений precision.
		:param recalls: Список значений recall.
		:return: Значение средней точности AP.
		"""
		precisions = [0] + precisions + [0]
		recalls = [0] + recalls + [1]

		for i in range(len(precisions) - 2, -1, -1):
			precisions[i] = max(precisions[i], precisions[i + 1])

		ap = 0
		for i in range(1, len(recalls)):
			ap += (recalls[i] - recalls[i - 1]) * precisions[i]
		return ap


def main():
	args = cli_arguments()

	# Инициализация класса и загрузка данных
	ap_calculator = AveragePrecisionCalculator(iou_threshold=args.iou_threshold)
	ap_calculator.load_groundtruths(args.groundtruth_path)
	ap_calculator.load_detections(args.detections_path)

	# Вычисление средней точности
	ap = ap_calculator.calculate_average_precision()

	# Печать результата
	if args.verbose:
		print(f"Groundtruths: {ap_calculator.groundtruths}")
		print(f"Detections: {ap_calculator.detections}")
	print(f"Average Precision (AP): {ap:.4f}")
