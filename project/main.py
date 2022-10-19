from time import time

from flask import Flask, render_template, request, redirect, url_for

# серые ипморты тут нужны, чтобы пикл не ругался: ему в основном файле нужны классы, которые он загружает
from engine import SearchEngine, BertSearchEngine, BM25SearchEngine, TfIdfSearchEngine

# создаем объект приложения фласка
app = Flask(__name__)

# задаем глобальные переменные для сайта - название и базовый адрес
title = 'Simple Search Engine'
url = 'http://127.0.0.1:5000/'

# задаем глобальные переменные для инициализации поисковика
PATH = 'data.jsonl'
DATA_PATH = 'data.pkl'
DATA_LINE = 80880

# инициализируем поисковик
engine = SearchEngine(PATH, DATA_PATH, DATA_LINE)


@app.route('/')
def index(site_title: str = title, baseurl: str = url) -> str:
    """
    Создает основную страницу сайта.
    :param site_title: название сайта.
    :param baseurl: адрес начальной страницы.
    :return: основная страница сайта.
    """
    page = render_template('index.html', page_title=None,
                           site_title=site_title, baseurl=baseurl)
    return page


@app.route('/search')
def search(site_title: str = title, baseurl: str = url) -> str:
    """
    Создает страницу с поиском.
    :param site_title: название сайта.
    :param baseurl: адрес начальной страницы.
    :return: страница с поиском.
    """
    page = render_template('search.html', page_title='Поиск', site_title=site_title, baseurl=baseurl)
    return page


@app.route('/find', methods=['get'])
def find_process(site_title: str = title, baseurl: str = url) -> str:
    """
    Ловит и обрабатывает запрос из формы поиска.
    :param site_title: название сайта.
    :param baseurl: адрес начальной страницы.
    :return: страница с формой поиска и результатами запроса.
    """
    if not request.args:
        return redirect(url_for('search'))
    begin_time = time()
    args = dict(request.args)
    args['res_line'] = int(args['res_line'])
    args['ans_line'] = int(args['ans_line'])

    res_ques, res_ans = engine.search(**args)
    search_time = round(time() - begin_time, 5)
    page = render_template('results.html', page_title='Поиск: результаты', site_title=site_title, baseurl=baseurl,
                           res_ques=res_ques, res_ans=res_ans, time=search_time, query=args['query'])
    return page


if __name__ == '__main__':
    engine.read_data()
    app.run()
