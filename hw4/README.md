# Инфопоиск. Домашняя работа №4
## Результаты оценки поиска
Считается доля запросов, для которых правильный ответ попал в топ-n релевантных ответов.

Top-n|Bert|Bm25
---|----|----
1|0.01236|0.03101
5|0.02059|0.0475
10|0.02734|0.05851
15|0.03244|0.0667
20|0.0364|0.07305
25|0.04011|0.07821
30|0.04271|0.08269
35|0.04554|0.08677
40|0.04812|0.09058
45|0.05053|0.09389
50|0.05288|0.09707
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
