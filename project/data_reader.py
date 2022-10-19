from tqdm import tqdm
import json

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2


class DataPreprocessor:
    def __init__(self):
        """
        Инициализации препроцессора.
        """
        self.cache = {}
        self.morph = pymorphy2.MorphAnalyzer()
        self.stops = stopwords.words('russian')

    def preprocess_text(self, text: str) -> str:
        """
        Препроцессинг: убираем стоп-слова, пунктуацию, приводим к нижнему регистру, лемматизируем.
        :param text: текст.
        :return: предобработанный текст.
        """
        res = []
        for el in word_tokenize(text):
            new_el = el.lower()
            if new_el not in self.cache:
                self.cache[new_el] = self.morph.parse(new_el)[0].normal_form
            word = self.cache[new_el]
            if word.isalpha() and word not in self.stops:
                res.append(word)
        return ' '.join(res)


class DataReader:
    def __init__(self, path: str, line: int = 50000, tq: bool = True):
        """
        Инициализируем класс чтения данных.
        :param path: путь к корпусу.
        :param line: кол-во элементов корпуса, которые требуется вернуть.
        :param tq: требуется ли прогрессбар.
        """
        self.path = path
        self.line = line
        self.tq = tq
        self.preprocessor = DataPreprocessor()

    @staticmethod
    def _process_answer(answer: dict) -> tuple[str, int]:
        """
        Получает ответы в необходимом формате.
        :param answer: ответ
        :return: ответ, его рейтинг
        """
        text = answer['text']
        value = answer['author_rating']['value']
        if value != '':
            value = int(value)
        else:
            value = 0
        return text, value

    def _process_item(self, item):
        """
        Обрабатывает один элемент из корпуса.
        :param item: элемент.
        :return: вопрос, предобработанный вопрос, ответы, предобработанные ответы.
        """
        ques = item['question']
        answers = item['answers']
        if len(answers) > 0:
            ques_prep = self.preprocessor.preprocess_text(ques)
            answers = [self._process_answer(ans) for ans in answers]
            answers = [i[0] for i in sorted(answers, key=lambda x: x[1], reverse=True)]
            answers_prep = [self.preprocessor.preprocess_text(ans) for ans in answers]
            return ques, ques_prep, answers, answers_prep

    def read_data(self):
        """
        Читает все данные из корпуса.
        :return: вопросы корпуса, отранжированные ответы на них, предобработанные вопросы, предобработанные ответы
        """
        count = 0
        tq = tqdm()
        results = []
        with open(self.path, encoding='utf-8') as f:
            s = f.readline()
            while (s != '') and (count < self.line):
                s = json.loads(s)
                res = self._process_item(s)
                if res:
                    results.append(res)
                    tq.update(1)
                    count += 1
                s = f.readline()
        return list(zip(*results))
