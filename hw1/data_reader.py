import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2


class DataReader:
    def __init__(self, path, prep=True):
        """
        Инициализируем класс чтения данных.
        :param path: путь к корпусу.
        :param prep: требуется ли препроцессинг.
        """
        self.path = path
        self.prep = prep
        self.morph = pymorphy2.MorphAnalyzer()
        self.stops = stopwords.words('russian')
        self.cash = {}

    def __preprocess_data(self, text: str) -> str:
        """
        Препроцессинг: убираем стоп-слова, пунктуацию, приводим к нижнему регистру, лемматизируем.
        :param text: текст.
        :return: предобработанный текст.
        """
        res = []
        for el in word_tokenize(text):
            new_el = el.lower()
            if new_el not in self.cash:
                self.cash[new_el] = self.morph.parse(new_el)[0].normal_form
            word = self.cash[new_el]
            if word.isalpha() and word not in self.stops:
                res.append(word)
        return ' '.join(res)

    def __read_data(self) -> tuple[str, str]:
        """
        Чтение данных.
        :return: название серии, текст серии.
        """
        for dr in os.listdir(self.path):
            for el in os.listdir(f'{self.path}/{dr}'):
                with open(f'{self.path}/{dr}/{el}', encoding='utf-8') as f:
                    text = f.read()
                    if self.prep:
                        text = self.__preprocess_data(text)
                    yield el, text

    def read_data(self) -> tuple[list, list]:
        """
        Читает все данные из корпсуа через ридер.
        :return: данные корпуса, список эпизодов.
        """
        corpus = []
        episodes = []
        reader = self.__read_data()
        for el in reader:
            episodes.append(el[0])
            corpus.append(el[1])
        return corpus, episodes
