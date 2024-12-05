import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Для прогресс-бара


def process_image(image_file, image_dir, output_dir, image_annotations, progress_bar):
	# Пропускаем не файлы изображений
	if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
		progress_bar.update(1)
		return

	# Извлечение номера изображения из названия файла
	try:
		image_number = os.path.splitext(image_file)[0].lstrip('0')
		if not image_number:
			image_number = 0
		else:
			image_number = int(image_number)
	except ValueError:
		print(f"Пропущен файл с некорректным именем: {image_file}")
		progress_bar.update(1)
		return

	if image_number not in image_annotations:
		progress_bar.update(1)
		return

	# Загрузка изображения
	image_path = os.path.join(image_dir, image_file)
	image = cv2.imread(image_path)
	if image is None:
		print(f"Не удалось загрузить изображение: {image_path}")
		progress_bar.update(1)
		return

	# Нанесение прямоугольников
	for label, x1, y1, x2, y2 in image_annotations[image_number]:
		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# Сохранение результата
	output_path = os.path.join(output_dir, image_file)
	cv2.imwrite(output_path, image)

	# Обновление прогресс-бара
	progress_bar.update(1)


def draw_rectangles(image_dir, annotation_file, num_images):
	if not os.path.exists(annotation_file):
		print(f"Файл разметки {annotation_file} не найден!")
		return

	# Чтение разметки из txt файла
	with open(annotation_file, 'r') as file:
		annotations = file.readlines()

	# Группировка аннотаций по номерам изображений
	image_annotations = {}
	for line in annotations:
		parts = line.strip().split()
		image_number = int(parts[0])
		label = parts[1]
		x1, y1, x2, y2 = map(int, parts[2:])
		if image_number not in image_annotations:
			image_annotations[image_number] = []
		image_annotations[image_number].append((label, x1, y1, x2, y2))

	# Директория для сохранения
	output_dir = os.path.join(image_dir, "annotated")
	os.makedirs(output_dir, exist_ok=True)

	# Получение списка файлов изображений
	image_files = sorted(os.listdir(image_dir))[:num_images]

	# Прогресс-бар
	with tqdm(total=len(image_files), desc="Обработка изображений") as progress_bar:
		# Многопоточность
		with ThreadPoolExecutor() as executor:
			futures = [
				executor.submit(process_image, image_file, image_dir, output_dir, image_annotations, progress_bar)
				for image_file in image_files
			]

			# Ожидание завершения всех потоков
			for future in futures:
				future.result()


if __name__ == '__main__':
	image_dir = "DLMini/data/imgs_track_09_0-2000"  # Замените на путь к папке с изображениями
	annotation_file = "DLMini/layout/track_09_0-2000.txt"  # Замените на путь к txt файлу с разметкой
	num_images = 1000  # Количество изображений для обработки

	draw_rectangles(image_dir, annotation_file, num_images)