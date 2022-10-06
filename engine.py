import pickle
from sklearn.feature_extraction.text import CountVectorizer
from data_reader import DataReader
from scipy import sparse
import numpy as np
import scipy as sp
import os


class SearchEngine:
    def __init__(self, path: str, data_path: str, data_line: int, res_line: int, ans_line: int) -> None:
        """
        Инициализация объекта класса поисковика: создает все необходимые атрибуты.
        :param path: путь к данным корпуса.
        :param data_path: путь к предобработанным данным с предыдущих запусков.
        :param data_line: кол-во элементов корпуса, которые используются в работе.
        :param res_line: кол-во элементов результата, которые надо выводить.
        :param ans_line: кол-во ответов на один вопрос, которые надо выводить.
        """
        self.path = path
        self.data_path = data_path
        self.reader = DataReader(self.path, line=data_line)
        self.res_line = res_line
        self.ans_line = ans_line
        self.exists = self._check_dir()
        print('Data reading...')
        self.questions, self.questions_prep, self.answers, self.vectorizer, self.vectorized_corpus = self._read_data()
        if self.vectorized_corpus is None:
            print('Corpus indexing...')
            self.process_corpus()
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
            questions, questions_prep, answers = self.reader.read_data()
            data = [np.array(questions, dtype=object), np.array(questions_prep, dtype=object),
                    np.array(answers, dtype=object), None, None]
        else:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def _save_data(self) -> None:
        """
        Сохраняет данные корпуса.
        """
        data = [self.questions, self.questions_prep, self.answers, self.vectorizer, self.vectorized_corpus]
        with open(self.data_path, 'wb') as f:
            pickle.dump(data, f)

    def process_corpus(self) -> None:
        """
        Векторизует корпуса с помощью метрики бм25.
        :return:
        """
        self.vectorizer = CountVectorizer()
        x = self.vectorizer.fit_transform(self.questions_prep)

        n = x.shape[0]
        nqi = np.unique(x.indices, return_counts=True)[1]
        idf = np.log((nqi - n - 0.5) / (nqi + 0.5) * -1)

        k = 2
        b = 0.75

        ld = (np.append(x.indptr[1:], 0) - x.indptr)[:-1]
        avgdl = np.mean(ld)

        xs, ys = x.nonzero()
        values = []
        for doc, word in zip(xs, ys):
            tf = x[doc, word]
            bm25 = idf[word] * tf * (k + 1) / (tf + k * (1 - b + b * ld[doc] / avgdl))
            values.append(bm25)
        self.vectorized_corpus = sparse.csr_matrix((values, (xs, ys)))

    def transform(self, corpus: list) -> sp.sparse:
        """
        Векторизует данные.
        :param corpus: корпус.
        :return: матрица count vectorizer
        """
        return self.vectorizer.transform(corpus)

    def compare(self, line: sp.sparse) -> sp.sparse:
        """
        Сравнивает матрицу запроса с матрицей всего корпуса.
        :param line: матрица запроса.
        :return: результат умножения матрицы корпуса на матрицу запроса.
        """
        return self.vectorized_corpus * line.reshape((-1, 1))

    def search(self, line: str) -> tuple[list, list]:
        """
        Основная функция поиска.
        :param line: запрос.
        :return: результаты, отсортированные в порядке убывания сходства.
        """
        line = self.reader.preprocess_data(line)
        idx_line = self.transform([line, ])[0]
        scores = self.compare(idx_line).toarray()
        sorted_scores_idx = np.argsort(scores, axis=0)[:-(self.res_line + 1):-1]
        res_ans = [el[:self.ans_line] for el in self.answers[sorted_scores_idx.ravel()]]
        res_ques = self.questions[sorted_scores_idx.ravel()]
        return res_ques, res_ans
