import test_AvPr as ap
import fake_detector as fakeD


if __name__ == '__main__':
	groundtruth_path = "DLMini/layout/track_09_0-2000.txt"
	detections_path = "fake_detections.txt"

	fakeD.generate(groundtruth_path, detections_path)

	# Инициализация класса и загрузка данных
	ap_calculator = ap.AveragePrecisionCalculator()
	ap_calculator.load_groundtruths(groundtruth_path)
	ap_calculator.load_detections(detections_path)

	# Вычисление средней точности
	ap = ap_calculator.calculate_average_precision()

	# Печать результата
	print(f"Average Precision (AP): {ap:.4f}")
