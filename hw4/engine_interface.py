# импорт отдельных поисковиков не используется в самой программе, но требуется для корректной работы пикл
from engine import SearchEngine, BertSearchEngine, BM25SearchEngine
import argparse


class TerminalParsers:
    def __init__(self) -> None:
        """
        Инициализируем нужные парсеры: внешний и внутренний
        """
        self.main_parser = None
        self.inner_parser = None
        self.get_main_parser()
        self.get_inner_parser()

    def get_main_parser(self) -> None:
        """
        Описываем основной парсер командной строки
        """
        self.main_parser = argparse.ArgumentParser(description='Really simple bm25 and bert search engine. '
                                                               'But it works, I hope :-)\n'
                                                               'Print -h inside this app to see the query format')
        self.main_parser.add_argument('-p', type=str, help='The path to corpus file in jsonl format. '
                                                           'Default = "data.jsonl". Considered if data path (-d) is '
                                                           'not valid or mentioned.', default='data.jsonl')
        self.main_parser.add_argument('-s', type=int, help='Amount of corpus lines processed. '
                                                           'Default = 80880 (full hw corpus).', default=80880)
        self.main_parser.add_argument('-d', type=str, help='Path to earlier processed data for search engine if exists.'
                                                           ' Otherwise it is used for data storage after corpora '
                                                           'processed. Default = "data.pkl"', default='data.pkl')

    def get_inner_parser(self) -> None:
        """
        Описываем внутренний парсер
        """
        self.inner_parser = argparse.ArgumentParser(description='Here you can print your query. Press Enter without '
                                                                'writing anything if you want to stop.')
        self.inner_parser.add_argument('query', nargs='*', type=str)
        self.inner_parser.add_argument('-l', type=int, help='Amount of results for each query. Default = 5.', default=5)
        self.inner_parser.add_argument('-a', type=int, help='Amount of answers for one question for each example from '
                                                            'results. Default = 5.', default=5)
        self.inner_parser.add_argument('-e', type=str, help='Engine to use: bert or bm25. Default = "bert"',
                                       default='bert')

    def parse_main_args(self) -> tuple[str, int, str]:
        """
        Получаем аргументы основного запуска через main_parser
        :return: путь к корпусу, кол-во строк для обработки, путь к сохраненным данным с предыдущего запуска
        """
        args = self.main_parser.parse_args()
        path = args.p
        data_line = args.s
        data_path = args.d
        return path, data_line, data_path

    def parse_inner_args(self, line: str) -> [bool, tuple[str, int, int, str]]:
        """
        Получаем аргументы запросов через внутренний парсер.
        :param line: строка запроса с парметрами.
        :return: запрос, кол-во результатов в вопросах, кол-во результатов в ответах, тип поиска.
        """
        line = line.split()

        if '-h' in line:
            self.inner_parser.print_help()
            return None

        else:
            args = self.inner_parser.parse_known_args(line)
            if isinstance(args, tuple):
                args = args[0]
            query = ' '.join(args.query)
            res_line = args.l
            ans_line = args.a
            eng_type = args.e
            return query, res_line, ans_line, eng_type


class SearchEngineTerminalInterface:
    def __init__(self) -> None:
        """
        Инициализация интерфейса
        """
        self.parser = TerminalParsers()
        path, data_line, data_path = self.parser.parse_main_args()
        self.engine = SearchEngine(path, data_path, data_line=data_line)
        self.engine.read_data()

    def run_engine(self) -> None:
        """
        Запуск поисковика
        """
        stop = False
        print(self.parser.main_parser.description)
        while not stop:

            # проверка ошибок осталась с предыдущей домашки, так как слабо верится в то, что оно самопочинилось
            # вот тут что-то идет не так в случайные моменты жизни, поэтому стоит обработка, но без фанатизма
            # есть подозрение, что зависит от каких-то штук, которые меняются от запуска к запуску,
            # поэтому если не получилось, то на выход.
            try:
                query = input('>>>\t')
            except UnicodeDecodeError:
                print('Sorry, something has gone wrong. Please, restart the application')
                break

            if query == '':
                stop = True

            else:
                inner = self.parser.parse_inner_args(query)
                if inner is not None:
                    query, res_line, ans_line, eng_type = inner
                    res_ques, res_ans = self.engine.search(query, eng_type, res_line, ans_line)
                    for i, el in enumerate(res_ques):
                        print(f'{i + 1}. {el}:')
                        for j, el1 in enumerate(res_ans[i]):
                            print(el1)
                    print('\n')


def main() -> None:
    """
    Основная функция запуска
    """
    engine = SearchEngineTerminalInterface()
    engine.run_engine()


if __name__ == '__main__':
    main()
