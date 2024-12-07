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

		Схема вычисления:
		1. Обнаруженные окаймляющие прямоугольники сортируются
		   в порядке убывания достоверности наличия в них объектов.
		2. Для каждого обнаруженного прямоугольника выполняется
		   поиск соответствия из разметки согласно условию IoU ≥ τ.
		3. Выполняется вычисление точности (Precision) и отклика (Recall).
		(4). Строится зависимость точности от отклика.
		5. Вычисляется площадь под графиком построенной зависимости (AP - Average Precision).

		Предположительно разметка детектора имеет следующий вид:
		0 CAR 0.77 232 128 290 168

		:return: Значение средней точности AP.
		"""
		# 1. Сортируем предсказания по достоверности
		all_detections = self._sort_detections_by_confidence()

		# 2. Поиск соответствия из разметки для каждого обнаруженного прямоугольника
		tp, fp, fn = 0, 0, sum(len(self.groundtruths.get(frame, [])) for frame in self.groundtruths)
		all_tp, all_fp, all_fn = [], [], []
		for frame_id, dets in all_detections.items():
			gts = self.groundtruths.get(frame_id, [])	# список всех прямоугольник для кадра
			tp_det, fp_det, fn_det = self._match_detections_to_groundtruth(dets, gts)
			tp += tp_det
			fp += fp_det
			fn -= tp_det
			all_tp.append(tp)
			all_fp.append(fp)
			all_fn.append(fn)

		# 3. Вычисляем precision и recall для всех точек
		precisions, recalls = self._calculate_precision_recall(all_tp, all_fp, all_fn)

		# 5. Вычисляем площадь под графиком зависимости точности от отклика
		return self._compute_ap(precisions, recalls)

	# ======= Приватные методы =======
	@staticmethod
	def _parse_annotations(file_path):
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
				bbox = list(map(float, bbox))	# Преобразуем str в float
				if frame_id not in annotations:
					annotations[frame_id] = []
				annotations[frame_id].append(bbox)
		return annotations

	@staticmethod
	def _parse_detections(file_path):
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

	@staticmethod
	def _calculate_iou(bbox1, bbox2):
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
			# среди всех прямоугольников истинной разметки ищем тот, с которым наибольшее значение iou
			for idx, gt in enumerate(groundtruths):
				iou = self._calculate_iou([x1, y1, x2, y2], gt)
				if iou > best_iou:
					best_iou = iou
					best_gt_idx = idx

			# Потенциальные проблемы с best_gt_idx not in matched
			# Возможна ситуация, когда для 2-х предсказаний значение iou будем максимальным с одним прямоугольником истинной разметки,
			# Допустим у первого iou = 0.7, у второго iou = 0.8.
			# Пусть первым обработали тот, у которого iou = 0.7. Следовательно, этот прямоугольник добавили в matched.
			# Далее обработали тот, у которого iou = 0.8. Однако его прямоугольник уже находится в matched, и сработает условие с fp += 1.

			# Суть в том, что мы ищем соответствия для истинной разметки. И если нашлось несколько
			# соответствий с best_iou >= self.iou_threshold, значит какое-то из них лишнее

			# Но есть ощущение, что могут быть проблемы
			if best_iou >= self.iou_threshold and best_gt_idx not in matched:
				tp += 1
				matched.add(best_gt_idx)
			else:
				# также сюда добавляются повторные детекции
				fp += 1

		fn = len(groundtruths) - len(matched)

		return tp, fp, fn

	def _sort_detections_by_confidence(self):
		"""
		Сортирует предсказания по достоверности.
		Предположительно разметка детектора имеет следующий вид:
		0 CAR 0.77 232 128 290 168

		:return: Словарь {frame_id: [отсортированный список предсказаний]}.
		"""
		sorted_detections = {}
		for frame, dets in self.detections.items():
			sorted_detections[frame] = sorted(dets, key=lambda x: -x[1])
		return sorted_detections

	def _calculate_precision_recall(self, all_tp, all_fp, all_fn):
		"""
		Вычисляет массивы precision (точности) и recall (отклика).

		:param all_tp: Список TP (true positives).
		:param all_fp: Список FP (false positives).
		:return: Списки значений precision и recall.
		"""
		count_gt = [len(self.groundtruths.get(frame, [])) for frame in self.groundtruths]
		count_det = [len(self.groundtruths.get(frame, [])) for frame in self.detections]

		precisions = []
		recalls = []
		for tp, fp, fn, gt, det in zip(all_tp, all_fp, all_fn, count_gt, count_det):
			if gt > 0 and det > 0:
				precision = tp / (tp + fp)
				recall = tp / (tp + fn)
			elif gt == 0 and det > 0:  # tp == 0 nad fp > 0 and fn == 0
				precision = 0
				recall = 1
			elif gt > 0 and det == 0:  # tp == 0 nad fp == 0 and fn > 0
				precision = 1
				recall = 0
			elif gt == 0 and det == 0:  # tp == 0 nad fp == 0 and fn == 0
				precision = 1
				recall = 1

			precisions.append(precision)
			recalls.append(recall)
		return precisions, recalls

	@staticmethod
	def _compute_ap(precisions, recalls):
		"""
		Вычисляет площадь под кривой зависимости precision от recall.

		:param precisions: Список значений precision.
		:param recalls: Список значений recall.
		:return: Значение средней точности AP.
		"""
		precisions = [1.0] + precisions + [-1]
		recalls = [0.0] + recalls + [recalls[-1]]

		last_prec = 1
		last_rec = 0
		ap = 0
		for i in range(len(precisions)):
			if last_prec <= precisions[i]:
				continue
			else:
				ap += (last_prec * (recalls[i] - last_rec))
				last_rec = recalls[i]
				last_prec = precisions[i]

		return ap