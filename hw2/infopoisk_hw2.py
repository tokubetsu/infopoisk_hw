from engine import SearchEngine
import argparse


def main() -> None:
    """
    Основная функция, которая собирает все остальные в полноценную домашку.
    """

    parser = argparse.ArgumentParser(description='Really, REALLY simple search engine. But it works, I hope :-)')
    parser.add_argument('-p', type=str, help='The path to "Friends" scripts. Default = "friends-data". Considered if '
                                             'data path is not valid or mentioned.',
                        default='friends-data')
    parser.add_argument('-l', type=int, help='Amount of results for each query. Default = 20. If -1 then full list '
                                             'of results will be presented.', default=20)
    parser.add_argument('-d', type=str, help='Path to earlier processed data for search engine if exists. Otherwise it '
                                             'is used for data storage after corpora processed.',
                        default='data.pkl')

    args = parser.parse_args()
    path = args.p
    line = args.l
    data_path = args.d

    engine = SearchEngine(path, data_path)
    stop = False
    while not stop:

        # вот тут что-то идет не так в случайные моменты жизни, поэтому стоит обработка, но без фанатизма
        # есть подозрение, что зависит от каких-то штук, которые меняются от запуска к запуску, так что если со второго
        # раза не получилось, то на выход.
        try:
            query = input('Type your query, please (press Enter without writing anything if you want to stop): ')
        except UnicodeDecodeError:
            query = input('Type your query, please (press Enter if you want to stop): ')
        finally:
            print('Sorry, something has gone wrong. Please, restart the application')
        if query == '':
            stop = True
        else:
            res = engine.search(query)
            if line == -1:
                line = len(res)
            for el in res[:line]:
                print(el)


if __name__ == '__main__':
    main()
