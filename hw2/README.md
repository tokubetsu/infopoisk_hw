# Инфопоиск. Домашняя работа №2
## Формат данных
Данные должны выглядеть следующим образом:
```
- Блок 1
  - Документ 1
  - Документ 2
  - ...
- Блок 2
- ...
```
В качестве индексов документов используются названия файлов с ними.
## Запуск
Запуск из файла `infopoisk_hw2.py` или через терминал.</br> 
Структура команды:</br>
```
infopoisk_hw2.py [options]

Options:

  -p  The path to "Friends" scripts. Default = "friends-data". Considered if data path (-d) is not valid or mentioned.
  
  -l  Amount of results for each query. Default = 20. If -1 then full list of results will be presented.
  
  -d  Path to earlier processed data for search engine if exists. Otherwise it is used for data storage after corpora processed.
```
## Формат выходных данных
Результат - отсортированный в поорядке релевантности список документов - выводится в консоль построчно:
```
- Документ 1
- Документ 2
- ...
```
