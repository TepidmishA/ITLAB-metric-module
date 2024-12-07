import test_AvPr as ap

if __name__ == '__main__':
	args = ap.cli_arguments()

	# Инициализация класса и загрузка данных
	ap_calculator = ap.AveragePrecisionCalculator(iou_threshold=args.iou_threshold)
	ap_calculator.load_groundtruths(args.groundtruth_path)
	ap_calculator.load_detections(args.detections_path)

	# Вычисление средней точности
	ap = ap_calculator.calculate_average_precision()

	# Печать результата
	if args.verbose:
		print(f"Groundtruths: {ap_calculator.groundtruths}")
		print(f"Detections: {ap_calculator.detections}")
	print(f"Average Precision (AP): {ap:.4f}")