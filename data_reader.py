import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2
from tqdm import tqdm
import _io


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
        self.morph = pymorphy2.MorphAnalyzer()
        self.stops = stopwords.words('russian')
        self.cache = {}

    def preprocess_data(self, text: str) -> str:
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

    def _read_data(self, f: _io.TextIOWrapper) -> tuple[str, str, str]:
        """
        Ридер, который читает данные из файла. Вложенная, потому что больше нигде особо не нужна.
        :param f: объект файла.
        :return: вопрос, предобработанный вопрос, все ответы на него в порядке убывания популярности.
        """

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

        s = f.readline()
        count = 0
        if self.tq:
            tq_run = tqdm(total=self.line)
        while (s != '') and (count < self.line):
            s = json.loads(s)
            question = s['question']
            answers = [_process_answer(answer) for answer in s['answers']]
            answers = sorted(answers, key=lambda x: x[1], reverse=True)
            if len(answers) > 0:
                question_prep = self.preprocess_data(question)
                count += 1
                if self.tq:
                    tq_run.update(1)
                yield question, question_prep, answers
            s = f.readline()

    def read_data(self) -> tuple[list, list, list]:
        """
        Читает все данные из корпсуа через ридер.
        :return: впоросы корпуса, предобработанные впоросы корпуса, отранжированные ответы на них
        """
        questions = []
        answers = []
        questions_prep = []
        f = open(self.path, encoding='utf-8')
        reader = self._read_data(f)
        for ques, ques_prep, answer in reader:
            questions.append(ques)
            questions_prep.append(ques_prep)
            answers.append(answer)
        f.close()
        return questions, questions_prep, answers
