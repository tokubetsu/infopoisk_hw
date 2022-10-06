from engine import SearchEngine
import argparse


def main() -> None:
    """
    Основная функция, которая собирает все остальные в полноценную домашку.
    """

    parser = argparse.ArgumentParser(description='Really, (not) REALLY simple bm25 search engine. But it works, '
                                                 'I hope :-)')
    parser.add_argument('-p', type=str, help='The path to corpus file in jsonl format. Default = "data.jsonl". '
                                             'Considered if data path (-d) is not valid or mentioned.',
                        default='data.jsonl')
    parser.add_argument('-l', type=int, help='Amount of results for each query. Default = 5.', default=5)
    parser.add_argument('-a', type=int, help='Amount of answers for one question for each example from results. '
                                             'Default = 5.',
                        default=5)
    parser.add_argument('-s', type=int, help='Amount of corpus lines processed. Default = 80880 (full hw corpus).',
                        default=80880)
    parser.add_argument('-d', type=str, help='Path to earlier processed data for search engine if exists. Otherwise it '
                                             'is used for data storage after corpora processed.',
                        default='data.pkl')

    args = parser.parse_args()
    path = args.p
    res_line = args.l
    data_line = args.s
    data_path = args.d
    ans_line = args.a

    engine = SearchEngine(path, data_path, data_line=data_line, res_line=res_line, ans_line=ans_line)
    stop = False
    while not stop:

        # проверка ошибок осталась с предыдущей домашки, так как слабо верится в то, что оно самопочинилось
        # вот тут что-то идет не так в случайные моменты жизни, поэтому стоит обработка, но без фанатизма
        # есть подозрение, что зависит от каких-то штук, которые меняются от запуска к запуску, так что если со второго
        # раза не получилось, то на выход.
        try:
            query = input('Type your query, please (press Enter without writing anything if you want to stop): ')
        except UnicodeDecodeError:
            print('Sorry, something has gone wrong. Please, restart the application')
            break
        if query == '':
            stop = True
        else:
            res_ques, res_ans = engine.search(query)
            for i, el in enumerate(res_ques):
                print(f'{i + 1}. {el}:')
                for j, el1 in enumerate(res_ans[i]):
                    print(f'\t{el1[0]} (likes: {el1[1]})')
            print('\n')


if __name__ == '__main__':
    main()
