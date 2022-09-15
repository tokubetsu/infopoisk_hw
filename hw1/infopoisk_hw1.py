import argparse
from data_reader import DataReader
from matrix_index import MatrixIndex
from dict_index import DictIndex


def main(line: int = 20, path: str = 'friends-data') -> None:
    """
    Основная функция, которая собирает все остальные в полноценную домашку.
    :param line: сколько примеров из выдачи по заданию мы хотим видеть.
    :param path: путь к данным корпуса.
    """

    parser = argparse.ArgumentParser(description='Working with "Friends" scripts."')
    parser.add_argument('-p', type=str, help='The path to "Friends" scripts. Default = "friends-data"')
    parser.add_argument('-l', type=int, help='Amount of examples for each question. Default = 20.')

    args = parser.parse_args()
    if args.p:
        path = args.p
    if args.l:
        line = args.l

    reader = DataReader(path, prep=True)
    corpus, episodes = reader.read_data()

    matrix = MatrixIndex(corpus)
    dct = DictIndex(corpus, episodes)

    mx_dict, mxd = dct.get_frequent()
    mx_matrix, mxm = matrix.get_frequent()
    print(f'most frequent:\nmatrix: {mx_matrix[0:line]}, {mxm}\tdict: {mx_dict[0:line]}, {mxd}', end='\n\n')

    mn_dict, mnd = dct.get_rare()
    mn_matrix, mnm = matrix.get_rare()
    print(f'least frequent:\nmatrix: {mn_matrix[0:line]}, {mnm}\ndict: {mn_dict[0:line]}, {mnd}', end='\n\n')

    iw_dict = dct.is_everywhere()
    iw_matrix = matrix.is_everywhere()
    print(f'are everywhere:\nmatrix: {iw_matrix[0:line]}\ndict: {iw_dict[0:line]}', end='\n\n')

    ch_matrix, chm = matrix.character_frequent()
    ch_dict, chd = dct.character_frequent()
    print(f'most frequent character:\nmatrix: {ch_matrix[0:line]}, {chm}\tdict: {ch_dict[0:line]}, {chd}',
          end='\n\n')


if __name__ == '__main__':
    main()
