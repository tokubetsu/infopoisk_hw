# Инфопоиск. Домашняя работа №3
## Формат входных данных
Данные должны быть в формате jsonl, где каждая строка имеет вид, как показано на примере:
```
{"question": 
    "случается с тобой ситуация неординарная, через время задумываешься, неужели это со мной произошло", 
 "comment": 
    "разве могло так случится, ты же особенный)))но нет и с тобой случается всякая нелепость, так как так получается в этой говённой жизни, что мы должны страдать душой а порой и телом, отчего нам такое прилетает....", 
 "sub_category": "relations", 
 "author": "悟", 
 "author_rating": 
     {
         "category": "Мастер", 
         "value": "1738"
     }, 
 "answers": [
     {
         "text": "Страдать или нет, люди сами себе выбирают сия \"творчество\", да и произошло значит так, и должно было быть.", 
         "author_rating": {"category": "Мыслитель", "value": "9931"}
     }, 
     {
         "text": "Dерьмо случается...)", 
         "author_rating": {"category": "Мудрец", "value": "11813"}
     }, 
     {
         "text": "наплюй", 
         "author_rating": {"category": "Мудрец", "value": "10611"}
     }], 
 "poll": []}
```
В качестве индексов документов используются формулировки вопросов.
## Запуск
Запуск из файла `engine_interface.py` или через терминал.</br> 
Структура команды запуска поисковика:</br>
```
infopoisk_hw3.py [options]

Options:

  -p  The path to corpus file in jsonl format. Default = "data.jsonl". Considered if data path (-d) is not valid or mentioned.
  
  -s  Amount of corpus lines processed. Default = 80880 (full hw corpus).
  
  -d  Path to earlier processed data for search engine if exists. Otherwise it is used for data storage after corpora processed. Default = "data.pkl"
```

Структура внутренней команды запроса:</br>
```
query [options]

Options:
  
  -l  Amount of results for each query. Default = 5.
  
  -a  Amount of answers for one question for each example from results. Default = 5.
  
  -e  Engine to use: bert or bm25. Default = "bert"
```

## Формат выходных данных
Результат - отсортированный в поорядке релевантности список вопросов и для каждого вопроса - отсортированный по кол-ву голосов список ответов.
Ввыводится в консоль построчно:
```
- Вопрос 1
  - Ответ 1
  - Ответ 2
  - ...
- Вопрос 2
  - Ответ 1
  - Ответ 2
  - ...
- ...
```
