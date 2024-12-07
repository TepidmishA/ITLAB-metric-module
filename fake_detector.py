import random


class FakeDetector:
    """
    Фейковый детектор, использующий истинную разметку для создания изменённых данных.
    """

    @staticmethod
    def read_ground_truth(input_file):
        """
        Читает файл с истинной разметкой.

        :param input_file: Путь к файлу с истинной разметкой.
        :return: Список строк с истинной разметкой.
        """
        with open(input_file, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    @staticmethod
    def modify_detection(line):
        """
        Модифицирует одну строку разметки, добавляя случайную точность и слегка изменяя координаты.

        :param line: Строка с разметкой вида "0 CAR x_min y_min x_max y_max".
        :return: Изменённая строка с добавленной точностью и изменёнными координатами.
        """
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"Некорректная строка разметки: {line}")

        frame_id = parts[0]
        label = parts[1]
        x_min, y_min, x_max, y_max = map(int, parts[2:])

        # Добавляем случайную точность от 0.7 до 0.99
        confidence = round(random.uniform(0.7, 0.99), 2)

        # Немного изменяем координаты (сдвиг на ±5 пикселей)
        delta = 5
        x_min = max(0, x_min + random.randint(-delta, delta))
        y_min = max(0, y_min + random.randint(-delta, delta))
        x_max = max(0, x_max + random.randint(-delta, delta))
        y_max = max(0, y_max + random.randint(-delta, delta))

        return f"{frame_id} {label} {confidence} {x_min} {y_min} {x_max} {y_max}"

    @staticmethod
    def generate_detections(input_file, output_file):
        """
        Генерирует фейковые детекции на основе истинной разметки и сохраняет их в файл.

        :param input_file: Путь к файлу с истинной разметкой.
        :param output_file: Путь к файлу для сохранения фейковых детекций.
        """
        ground_truth = FakeDetector.read_ground_truth(input_file)
        with open(output_file, 'w') as f:
            for line in ground_truth:
                modified_line = FakeDetector.modify_detection(line)
                f.write(modified_line + '\n')
        print(f"Фейковые детекции сохранены в файл: {output_file}")


def generate(input_file, output_file):
    FakeDetector.generate_detections(input_file, output_file)
