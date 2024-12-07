import average_precision

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
						required=True)

	parser.add_argument("--detections", "-d",
						help="Path to the detections file.",
						type=str,
						dest="detections_path",
						required=True)

	parser.add_argument("--iou", "-iou",
						help="Intersection over Union (IoU) threshold (default: 0.5).",
						type=float,
						dest="iou_threshold",
						required=False,
						default=0.5)

	return parser.parse_args()


if __name__ == '__main__':
	args = cli_arguments()

	# Инициализация класса и загрузка данных
	ap_calculator = average_precision.AveragePrecisionCalculator()
	ap_calculator.load_groundtruths(args.groundtruth_path)
	ap_calculator.load_detections(args.detections_path)

	# Вычисление средней точности
	ap = ap_calculator.calculate_average_precision()

	# Печать результата
	print(f"Average Precision (AP): {ap:.4f}")
