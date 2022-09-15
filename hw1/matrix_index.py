from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class MatrixIndex:
    def __init__(self, corpus: list) -> None:
        """
        Инициализируем класс работы с индексом-матрицей.
        :param corpus: тексты корпуса.
        """
        self.index, self.words = self.get_index(corpus)
        self.freq = self.get_freq(self.index)
        self.chars = {'Моника': ['Моника', 'Мон'], 'Рэйчел': ['Рэйчел', 'Рейч'],
                      'Чендлер': ['Чендлер', 'Чэндлер', 'Чен'], 'Фиби': ['Фиби', 'Фибс'], 'Росс': ['Росс', ],
                      'Джоуи': ['Джоуи', 'Джои', 'Джо']}

    @staticmethod
    def get_index(corpus: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Получает матрицу-индекс.
        :param corpus: данные.
        :return: матрица-индекс, список слов.
        """
        vectorizer = CountVectorizer(analyzer='word')
        x = vectorizer.fit_transform(corpus)
        return x.toarray(), vectorizer.get_feature_names_out()

    @staticmethod
    def get_freq(matrix: np.ndarray) -> np.ndarray:
        """
        Получает матрицу частот слов.
        :param matrix: матрица-индекс.
        :return: None
        """
        temp = np.sum(matrix, axis=0, keepdims=True).reshape((-1,))
        return temp

    def get_rare(self) -> tuple[list, int]:
        """
        Получает список наименее частотных слов для матрицы.
        :return: список наименее частотных слов, минимальная частота.
        """
        mn = np.amin(self.freq)
        ws = self.words[np.where(self.freq == mn)[0]]
        return ws.tolist(), mn

    def get_frequent(self) -> tuple[list, int]:
        """
        Получает наиболее частотные слова (слово) для матрицы.
        :return:
        """
        mx = np.amax(self.freq)
        ws = self.words[np.where(self.freq == mx)[0]]
        return ws.tolist(), mx

    def is_everywhere(self) -> list:
        """
        Получает слова, которые есть во всех текстах для матрицы.
        :return: список слов, встречающихся во всех текстах.
        """
        idx = np.where(np.amin(self.index, axis=0) != 0)
        return self.words[idx].tolist()

    def characters(self) -> dict:
        """
        Получается словарь частот героев для матрицы.
        :return: словарь частот персонажей.
        """
        res = {}
        for char in self.chars:
            res[char] = 0
            for word in self.chars[char]:
                word_el = word.lower()
                temp = self.freq[np.where(self.words == word_el)]
                if temp.shape[0] > 0:
                    res[char] += temp[0]
        return res

    def character_frequent(self) -> tuple[list, int]:
        """
        Находит наиболее популярного персонажа (персонажей, если таких несколько).
        :return: самые популярные персонажи, частота.
        """
        char = self.characters()
        mx = max(char.values())
        res = []
        for el in char:
            if char[el] == mx:
                res.append(el)
        return res, mx
