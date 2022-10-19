from tqdm import tqdm
import os
import pickle

import numpy as np
import scipy as sp
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import torch
from transformers import AutoTokenizer, AutoModel
import transformers.modeling_outputs

from data_reader import DataReader


class BertSearchEngine:
    def __init__(self, vectorized_corpus: [None, torch.Tensor] = None, data: [None, list] = None) -> None:
        """
        Инициализация поисковика с берт-векторизацией.
        :param vectorized_corpus: векторизованный корпус.
        :param data: данные корпуса для векторизации (используется, если нет предыдущего)
        """
        self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
        self.vectorizer = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
        if vectorized_corpus is not None:
            self.vectorized_corpus = vectorized_corpus
        else:
            if data is not None:
                self.process_corpus(data)
            else:
                self.vectorized_corpus = None

    @staticmethod
    def mean_pooling(model_output: transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
                     attention_mask: torch.Tensor):
        """
        Считает эмбеддинг предложения, взято с сайте hugging face.
        :param model_output: выход модели для данного предложения.
        :param attention_mask: маска внимания.
        :return: вектор предложения.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_batch_embeddings(self, sentences):
        """
        Считает вектора для батча текстов.
        :param sentences: тексты.
        :return: эмбеддинги.
        """
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = self.vectorizer(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    @staticmethod
    def normalize_vectors(vec: torch.Tensor):
        """
        Нормализует вектора так, чтобы их потом можно было просто перемножать для поиска сходства.
        :param vec: матрица с эмбеддингами.
        :return: нормализованные вектора.
        """
        return torch.div(vec, torch.sqrt(torch.sum(torch.pow(vec, 2), dim=1)).reshape((-1, 1)).expand(vec.shape))

    def process_corpus(self, corpus: list, gap: int = 100, save: bool = True) -> [None, torch.Tensor]:
        """
        Обрабатывает весь корпус по частям, чтобы можно было отслеживать прогресс. Все равно считалось на процессоре.
        :param corpus: корпус для векторизации.
        :param gap: размер батча.
        :param save: сохранять или выдавать наружу.
        :return: либо ничего, либо векторизованный корпус.
        """
        res = []
        for i in tqdm(range(0, len(corpus), gap)):
            temp = corpus[i:i + gap]
            temp_embeddings = self.get_batch_embeddings(temp)
            res.append(temp_embeddings)
        if save:
            self.vectorized_corpus = self.normalize_vectors(torch.cat(res))
        else:
            return self.normalize_vectors(torch.cat(res))

    def transform(self, corpus: [list, np.ndarray]) -> torch.Tensor:
        """
        Векторизует данные.
        :param corpus: корпус.
        :return: нормализованные вектора для корпуса.
        """
        return self.normalize_vectors(self.get_batch_embeddings(corpus))

    def compare(self, line: torch.Tensor) -> torch.Tensor:
        """
        Сравнивает матрицу запроса с матрицей всего корпуса.
        :param line: матрица запроса.
        :return: результат умножения матрицы корпуса на матрицу запроса.
        """
        return torch.mm(self.vectorized_corpus, torch.transpose(line, 0, 1))


class BM25SearchEngine:
    def __init__(self, vectorizer=None, vectorized_corpus=None, data=None):
        """
        Инициализация поисковика с бм25-векторизацией.
        :param vectorizer: векторайзер.
        :param vectorized_corpus: обработанный корпус.
        :param data: корпус для векторизации (используется, если нет предыдущего).
        """
        if vectorizer is not None:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = CountVectorizer()

        if vectorized_corpus is not None:
            self.vectorized_corpus = vectorized_corpus
        else:
            if data is not None:
                self.process_corpus(data)
            else:
                self.vectorized_corpus = None

    def process_corpus(self, data) -> None:
        """
        Векторизует корпус с помощью метрики бм25.
        :param data: данные для векторизации.
        """
        x = self.vectorizer.fit_transform(data)

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

    def transform(self, corpus: [list, np.ndarray]) -> sp.sparse:
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
        return np.dot(self.vectorized_corpus, line.transpose())


class TfIdfSearchEngine:
    def __init__(self, vectorizer=None, vectorized_corpus=None, data=None):
        """
        Инициализация поисковика с tfidf-векторизацией.
        :param vectorizer: векторайзер.
        :param vectorized_corpus: обработанный корпус.
        :param data: корпус для векторизации (используется, если нет предыдущего).
        """
        if vectorizer is not None:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = TfidfVectorizer()

        if vectorized_corpus is not None:
            self.vectorized_corpus = vectorized_corpus
        else:
            if data is not None:
                self.vectorizer.fit(data)
                self.process_corpus(data)
            else:
                self.vectorized_corpus = None

    def process_corpus(self, data, out=False) -> [None, sp.sparse]:
        """
        Векторизует корпус с помощью метрики tf-idf.
        :param data: данные для векторизации.
        :param out:
        """
        x = self.vectorizer.transform(data)
        xs, ys = x.nonzero()
        denominator = np.sqrt(np.array(x.power(2).sum(axis=1)))
        values = []
        for doc, word in zip(xs, ys):
            values.append((x[doc, word] / denominator[doc])[0])
        if not out:
            self.vectorized_corpus = sparse.csr_matrix((values, (xs, ys)))
        else:
            return sparse.csr_matrix((values, (xs, ys)), shape=(len(data), self.vectorized_corpus.shape[1]))

    def transform(self, corpus: [list, np.ndarray]) -> sp.sparse:
        """
        Векторизует данные.
        :param corpus: корпус.
        :return: матрица tf-idf
        """
        return self.process_corpus(corpus, out=True)

    def compare(self, line: sp.sparse) -> sp.sparse:
        """
        Сравнивает матрицу запроса с матрицей всего корпуса.
        :param line: матрица запроса.
        :return: результат умножения матрицы корпуса на матрицу запроса.
        """
        return np.dot(self.vectorized_corpus, line.transpose())


class SearchEngine:
    def __init__(self, path: str, data_path: str, data_line: int) -> None:
        """
        Инициализация объекта класса поисковика: создает все необходимые атрибуты.
        :param path: путь к данным корпуса.
        :param data_path: путь к предобработанным данным с предыдущих запусков.
        :param data_line: кол-во элементов корпуса, которые используются в работе.
        """
        self.path = path
        self.data_path = data_path
        self.exists = self._check_dir()
        self.reader = DataReader(self.path, line=data_line)

        self.questions = []
        self.questions_prep = []
        self.answers = []
        self.bert_engine = None
        self.bm25_engine = None
        self.tfidf_engine = None

    def _check_dir(self) -> bool:
        """
        Проверяет, есть ли в директории сохраненные ранее данные.
        :return: индикатор наличия данных
        """
        return os.path.isfile(self.data_path)

    def read_data(self) -> None:
        """
        Читает данные либо из предсохраненного, либо из корпуса.
        """
        if not self.exists:
            self.questions, self.questions_prep, self.answers, _ = self.reader.read_data()
            self.tfidf_engine = TfIdfSearchEngine(data=list(self.questions_prep))
            self.bm25_engine = BM25SearchEngine(data=list(self.questions_prep))
            self.bert_engine = BertSearchEngine(data=list(self.questions))
            self.questions = np.array(self.questions, dtype=object)
            self.questions_prep = np.array(self.questions_prep, dtype=object)
            self.answers = np.array(self.answers, dtype=object)

            self._save_data()
        else:
            with open(self.data_path, 'rb') as f:
                self.questions, self.questions_prep, self.answers, bert_vectors, self.bm25_engine, \
                    self.tfidf_engine = pickle.load(f)
                self.bert_engine = BertSearchEngine(vectorized_corpus=bert_vectors)

    def _save_data(self) -> None:
        """
        Сохраняет данные корпуса.
        """
        with open(self.data_path, 'wb') as f:
            pickle.dump([self.questions, self.questions_prep, self.answers, self.bert_engine.vectorized_corpus,
                         self.bm25_engine, self.tfidf_engine], f)

    def search(self, query: str, mode: str = 'bert', res_line: int = 5, ans_line: int = 5) -> [tuple[list, list], str]:
        """
        Основная функция поиска.
        :param query: запрос.
        :param mode: тип индексации корпуса для поиска.
        :param res_line: кол-во элементов результата, которые надо выводить.
        :param ans_line: кол-во ответов на один вопрос, которые надо выводить.
        :return: результаты, отсортированные в порядке убывания сходства.
        """
        if mode == 'bm25':
            query = self.reader.preprocessor.preprocess_text(query)
            engine = self.bm25_engine
        elif mode == 'bert':
            engine = self.bert_engine
        elif mode == 'tf-idf':
            engine = self.tfidf_engine
        else:
            return f'There is no engine called {mode}'

        idx_line = engine.transform([query, ])
        scores = engine.compare(idx_line)
        if isinstance(scores, torch.Tensor):
            scores = np.array(scores)
        else:
            scores = scores.toarray()
        sorted_scores_idx = np.argsort(scores, axis=0)[:-(res_line + 1):-1]
        res_ans = [el[:ans_line] for el in self.answers[sorted_scores_idx.ravel()]]
        res_ques = self.questions[sorted_scores_idx.ravel()]
        return res_ques, res_ans
