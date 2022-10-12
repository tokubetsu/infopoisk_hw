from engine import BertSearchEngine, BM25SearchEngine
from data_reader import DataReader
from tqdm import tqdm
import numpy as np
import pickle
import torch
import os


class EvaluationDataReader:
    def __init__(self, path: str, data_path: str, data_line: int) -> None:
        """
        Инициализация класса, который читает и процессит данные специально для evaluation, так как алгоритм самого
        поиска работает с другой логикой: там используется индексация вопросов, а не ответов.
        :param path: путь к данным корпуса.
        :param data_path: путь, куда надо сохранить предобработанные данные.
        :param data_line: кол-во элементов корпуса, которые надо обработать.
        """
        self.data_path = data_path
        self.reader = DataReader(path, data_line)
        try:
            os.mkdir(data_path)
        except FileExistsError:
            pass
        self.questions = None
        self.questions_prep = None
        self.answers = None
        self.answers_prep = None

        self.bert_engine = None
        self.bm25_engine = None
        self.questions_vectors_bert = None
        self.questions_vectors_bm25 = None
        self.answers_vectors_bert = None
        self.answers_vectors_bm25 = None

    def read_data(self) -> None:
        """
        Читает данные и либо запускает обработку, либо просто достает уже обработанное.
        """
        if 'texts.pkl' not in os.listdir(self.data_path):
            self.questions, self.questions_prep, self.answers, self.answers_prep = self.reader.read_data()

            self.answers, self.answers_prep = list(zip(*[(self.answers[i][0],
                                                          self.answers_prep[i][0]) for i in range(len(self.answers))]))
            with open(os.path.join(self.data_path, 'texts.pkl'), 'wb') as f:
                pickle.dump([self.questions, self.questions_prep, self.answers, self.answers_prep], f)
        else:
            with open(os.path.join(self.data_path, 'texts.pkl'), 'rb') as f:
                self.questions, self.questions_prep, self.answers, self.answers_prep = pickle.load(f)

        keys = ['answers_vectors_bm25', 'questions_vectors_bm25', 'answers_vectors_bert', 'questions_vectors_bert']
        temp_keys = set([key + '.pkl' for key in keys])

        if not temp_keys.issubset(set(os.listdir(self.data_path))):
            self.vectorize()

        for key in keys:
            with open(os.path.join(self.data_path, key + '.pkl'), 'rb') as f:
                setattr(self, key, pickle.load(f))

    def save_data(self, name, data) -> None:
        """
        Сохраняет обработанные данные в файл.
        :param name: имя файла без расширения.
        :param data: данные.
        """
        print(f'Saving {name}')
        with open(os.path.join(self.data_path, name + '.pkl'), 'wb') as f:
            pickle.dump(data, f)

    def vectorize(self) -> None:
        """
        Векторизует корпус.
        """
        self.bm25_engine = BM25SearchEngine(data=list(self.answers_prep))
        self.save_data('answers_vectors_bm25', self.bm25_engine.vectorized_corpus)
        self.questions_vectors_bm25 = self.bm25_engine.transform(list(self.questions_prep))
        self.save_data('questions_vectors_bm25', self.questions_vectors_bm25)

        self.bert_engine = BertSearchEngine(data=list(self.answers))
        self.save_data('answers_vectors_bert', self.bert_engine.vectorized_corpus)
        self.questions_vectors_bert = self.bert_engine.process_corpus(list(self.questions), save=False)
        self.save_data('questions_vectors_bert', self.questions_vectors_bert)


class EngineEvaluator:
    def __init__(self, path: str, data_path: str, data_line: int, name_eval: str = 'eval.pkl') -> None:
        """
        Инициализация класса, которые делает оценку поиска. Сделала в итоге так, потому что оно позволяет весь корпус
        обработать.
        :param path: путь к данным корпуса.
        :param data_path: путь к папке со всеми данными оценки.
        :param data_line: кол-во элементов корпуса, которые надо обработать.
        :param name_eval: имя для файла с данными по оценке.
        """
        self.data_path = data_path
        self.name_eval = name_eval
        self.reader = EvaluationDataReader(path, data_path, data_line)
        self.reader.read_data()
        self.bert_engine = BertSearchEngine(vectorized_corpus=self.reader.answers_vectors_bert)
        self.bm25_engine = BM25SearchEngine(vectorized_corpus=self.reader.answers_vectors_bm25)
        self.bert_res = []
        self.bm25_res = []

    @staticmethod
    def search_one_engine(i: int, query: [np.ndarray, torch.Tensor],
                          engine: [BertSearchEngine, BM25SearchEngine]) -> int:
        """
        Получает запрос и номер текста, чтобы найти на каком месте расположен правильный ответ. Логика такая: матрицы
        запросов и ответов соотнесены по позициям (пятый ответ - ответ на пятый вопрос), поэтому в идеале i-тая
        позиция должна оказываться первой, но это в идеале. Тут мы просто возвращаем позицию в рейтинге релевантности.
        :param i: номер строки запроса.
        :param query: вектор запроса.
        :param engine: поисковик для выполнения запроса.
        :return: позиция в рейтинге релевантности.
        """
        scores = engine.compare(query)
        if isinstance(scores, torch.Tensor):
            scores = np.array(scores)
        else:
            scores = scores.toarray()
        sorted_scores_idx = np.argsort(scores, axis=0)[::-1]
        pos = np.where(sorted_scores_idx == i)[0][0]
        return pos

    def get_poses(self):
        """
        Получает для всех запросов в корпусе позицию их ответа в рейтинге релевантных ответов.
        """
        for i, query in tqdm(enumerate(self.reader.questions_vectors_bert), total=len(self.reader.questions)):
            self.bert_res.append(self.search_one_engine(i, query, self.bert_engine))
            query = self.reader.questions_vectors_bm25[i]
            self.bm25_res.append(self.search_one_engine(i, query, self.bm25_engine))

    def evaluate(self, line: int = 5):
        """
        Внешняя функция оценки качества поиска.
        :param line: в топ сколько мы ищем данный элемент.
        :return: результаты для берта и бм25.
        """
        if self.name_eval not in os.listdir(self.data_path):
            self.get_poses()
            with open(os.path.join(self.data_path, self.name_eval), 'wb') as f:
                pickle.dump([self.bm25_res, self.bert_res], f)
        else:
            with open(os.path.join(self.data_path, self.name_eval), 'rb') as f:
                self.bm25_res, self.bert_res = pickle.load(f)

        self.bm25_res = np.array(self.bm25_res)
        res_bm25 = self.bm25_res[self.bm25_res <= line].shape[0] / self.bm25_res.shape[0]

        self.bert_res = np.array(self.bert_res)
        res_bert = self.bert_res[self.bert_res <= line].shape[0] / self.bert_res.shape[0]
        return res_bert, res_bm25


def main() -> None:
    """
    Основная функция, которая запускает оценку.
    """
    evaluator = EngineEvaluator('data.jsonl', 'evaluation_data', 80880)
    with open('evaluation_data/res.txt', 'a', encoding='utf-8') as f:
        for line in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            print(line)
            res = evaluator.evaluate(line=line)
            f.write(f'top-{line}:\n')
            f.write(f'bert:\t{res[0]}\n')
            f.write(f'bm25:\t{res[1]}\n\n')


if __name__ == '__main__':
    main()
