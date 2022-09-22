import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from data_reader import DataReader
import scipy as sp
import os


class SearchEngine:
    def __init__(self, path: str, data_path: str) -> None:
        """
        Инициализация объекта класса поисковика: создает все необходимые атрибуты.
        :param path: путь к данным корпуса.
        :param data_path: путь к предобработанным данным с предыдущих запусков.
        """
        self.path = path
        self.data_path = data_path
        self.reader = DataReader(self.path)
        self.exists = self._check_dir()
        self.corpus, self.episodes, self.vectorizer, self.vectorized_corpus = self._read_data()
        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer()
            self.vectorized_corpus = self.vectorizer.fit_transform(self.corpus)
        self._save_data()

    def _check_dir(self) -> bool:
        """
        Проверяет, есть ли в директории сохраненные ранее данные.
        :return: индикатор наличия данных
        """
        return os.path.isfile(self.data_path)

    def _read_data(self) -> list:
        """
        Читает данные либо из предсохраненного, либо из корпуса.
        :return: список из корпуса, эпизодов, (векторайзера и векторизованного корпуса)
        """
        if not self.exists:
            data = [*self.reader.read_data(), False, False]
        else:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def _save_data(self) -> None:
        """
        Сохраняет данные корпуса.
        """
        data = [self.corpus, self.episodes, self.vectorizer, self.vectorized_corpus]
        with open(self.data_path, 'wb') as f:
            pickle.dump(data, f)

    def transform(self, corpus: list) -> sp.sparse:
        """
        Векторизует данные.
        :param corpus: корпус.
        :return: матрица tf-idf
        """
        return self.vectorizer.transform(corpus)

    def compare(self, line: sp.sparse) -> sp.sparse:
        """
        Сравнивает матрицу запроса с матрицей всего корпуса.
        :param line: матрица запроса.
        :return: результат умножения матрицы корпуса на матрицу запроса.
        """
        return self.vectorized_corpus * line.reshape((-1, 1))

    def search(self, line: str) -> list:
        """
        Основная функция поиска.
        :param line: запрос.
        :return: результаты, отсортированные в порядке убывания сходства.
        """
        line = self.reader.preprocess_data(line)
        idx_line = self.transform([line, ])[0]
        res = self.compare(idx_line)
        return [el[1] for el in sorted(enumerate(self.episodes), key=lambda x: res[x[0]], reverse=True)]
