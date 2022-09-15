class DictIndex:
    def __init__(self, corpus: list, episodes: list) -> None:
        """
        Инициализируем класс работы с индексом-словарем.
        :param corpus: тексты корпуса.
        :param episodes: список эпизодов.
        """
        self.index = self.get_index(episodes, corpus)
        self.freq = self.get_freq(self.index)
        self.chars = {'Моника': ['Моника', 'Мон'], 'Рэйчел': ['Рэйчел', 'Рейч'],
                      'Чендлер': ['Чендлер', 'Чэндлер', 'Чен'], 'Фиби': ['Фиби', 'Фибс'], 'Росс': ['Росс', ],
                      'Джоуи': ['Джоуи', 'Джои', 'Джо']}

    @staticmethod
    def get_index(episodes: list, corpus: list) -> dict:
        """
        Получает словарь-индекс.
        :param episodes: список эпизодов.
        :param corpus: корпус.
        :return: словарь-индекс.
        """
        dct = {}
        for i, el in enumerate(episodes):
            for j, word in enumerate(corpus[i].split(' ')):
                if word not in dct:
                    dct[word] = {episode: 0 for episode in episodes}
                dct[word][el] += 1
        return dct

    @staticmethod
    def get_freq(ind_dict: dict) -> dict:
        """
        Получает словарь частот слов.
        :param ind_dict: словарь-индекс.
        :return: словарь частот слов.
        """
        dct = {}
        for word in ind_dict:
            for episode in ind_dict[word]:
                if word not in dct:
                    dct[word] = 0
                dct[word] += ind_dict[word][episode]
        return dct

    def get_frequent(self) -> tuple[list, int]:
        """
        Получает наиболее частотные слова (слово) для словаря.
        :return: список наиболее частотных слов, максимальная частота.
        """
        lst = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
        mx = lst[0][1]
        res = []
        for el in lst:
            if el[1] == mx:
                res.append(el[0])
            else:
                return res, mx
        return res, mx

    def get_rare(self) -> tuple[list, int]:
        """
        Получает список наименее частотных слов для словаря.
        :return: список наименее частотных слов, минимальная частота.
        """
        mn = min(self.freq.values())
        res = [el for el in self.freq if self.freq[el] == mn]
        return res, mn

    def is_everywhere(self) -> list:
        """
        Получает слова, которые есть во всех текстах для словаря.
        :return: список слов, встречающихся во всех текстах.
        """
        res = []
        for word in self.index:
            in_all = True
            for episode in self.index[word]:
                if self.index[word][episode] == 0:
                    in_all = False
            if in_all:
                res.append(word)
        return res

    def characters(self) -> dict:
        """
        Получает словарь частот героев для словаря.
        :return: словарь частот персонажей.
        """
        res = {}
        for char in self.chars:
            res[char] = 0
            for word in self.chars[char]:
                word_el = word.lower()
                res[char] += self.freq.get(word_el, 0)
        return res

    def character_frequent(self) -> tuple[list, int]:
        """
        Находит наиболее популярного персонажа (персонажей, если таких несколько).
        :return: самые популярные персонажи, частота.
        """
        char = self.characters()
        mx = max(char.values())
        res = []
        for el in char:
            if char[el] == mx:
                res.append(el)
        return res, mx
