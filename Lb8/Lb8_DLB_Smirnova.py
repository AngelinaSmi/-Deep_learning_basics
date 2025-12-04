# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:35:35 2025

@author: ПользовательHP
"""

# Для установки библиотеки воспользуйтесь командой
# !conda install conda-forge::transformers
# или
# !pip install transformers

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TLS_VERIFY"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda"
)



import dataclasses
print(dataclasses.__file__)

import torch

print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Название GPU:", torch.cuda.get_device_name(0))
    print("Количество GPU:", torch.cuda.device_count())

# Любая LLM работает с токенами, поэтому нам нужна не только модель, но и токенизатор
from transformers import AutoModelForCausalLM, AutoTokenizer

# Зададим название модели из репозитория huggingface
# список доступных моделей можно посмотреть по ссылке https://huggingface.co/models
# будем использовать модель Qwen или YandexGPT

model_name = "Qwen/Qwen2.5-3B-Instruct"
#model_name = "yandex/YandexGPT-5-Lite-8B-instruct"

# В названии модели обычно указываются - версия (2,5) количество параметров (7 млрд.)
# этап обучения или предназначение (Base, Instruct, Code, Thinking и т.п.)
# размер контекстного окна (1 млн.)
# также могут указываться параметры квантования (FP8, GGUF, ) и другие характеристики.


# Загружаем предобученную можель
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda"
)

# Загружаем предобученный токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загружаем текстовый файл преобразуя его в строку
file = 'ENG_article.txt'
with open(file, 'r', encoding='cp1251') as file:
    data = file.read().replace('\n', ' ')

print(len(data))

# Посмотрим выдержку из файла
print(data[90:160])

# Посчитаем количество слов в файле чтобы примерно сопоставить с контекстным окном
num_words = len(data.split())
print(num_words)

# Для общения с LLM нужно составить промпт
# он будет состоять из системной части - наши инструкции что нужно сделать
# и пользовательской части - текста файла
messages = [
    {"role": "system", "content": "Суммаризируй текст. В каком году была обозначена проблема взрывающихся градиентов? Ответ давай на русском языке."},
    {"role": "user", "content": data[0:8291]} # берем всё
]

# Для YandexGPT инструкции пишутся от имени пользователя
#messages = [
#    {"role": "user", "content": "Суммаризируй текст. Определи жанр текста, выдели основную информацию. Ответ давай на русском языке."+data[0:5000]} # берем только часть чтобы долго не ждать генерацию
#]


# Метод apply_chat_template() используется для форматирования сообщений в одну строку, формат
# которой ориентирова на использование чат-ориентированных языковых моделей.
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# В текст добавляются специальные  токены-метки для указания структуры разговора.
print(text[0:160])

# Токенизатор разбивает текст на токены.
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# input_ids — токены в числовом виде, которые представляют собой уникальный номер (ID) из словаря модели.
# attention_mask — задает маску, которая показывает позиции реальных токенов и дополнений (padding) или служебных токенов.
# Модели обрабатывают батчи текстов одинаковой длины. Чтобы выровнять длину реальных текстов короткие тексты дополняются специальным токеном [PAD].
# Таким токенам в attention_mask соответствуют нули, чтобы модель их не обрабатывала.
print(model_inputs)

# Токенизированный текст подаем в модель.
# max_new_tokens - задает максимальное число генерируемых в ответе токенов.
# Можно указать и другие параметры:
# temperature - регулирует "случайность" в выборе следующего токена (<1 — более предсказуемо, >1 — более хаотично).
# top_k - ограничивает выбор следующего токена K наиболее вероятными
# repetition_penalty - штраф за повторяющиеся токены (>1 - штраф, 1 - без штрафа).
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
#    temperature=0.9,
#    top_k=50,
#    repetition_penalty=1.2
)

# Так как модель возвращает и промпт и сгенерированные токены, выделяем только ответ.
generated_ids_ = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Преобразуем ID токенов обратно в слова.
response = tokenizer.batch_decode(generated_ids_, skip_special_tokens=True)[0]

# Смотрим что получилось.
print(response)


########################################2222222222222222222222222222##################################################
message2 = [
    {"role": "system", "content": "Кто в 1891 году разработал метод уничтожающей производной? Ответ давай на русском языке."},
    {"role": "user", "content": data[0:8291]} # берем всё
]

# Для YandexGPT инструкции пишутся от имени пользователя
#messages = [
#    {"role": "user", "content": "Суммаризируй текст. Определи жанр текста, выдели основную информацию. Ответ давай на русском языке."+data[0:5000]} # берем только часть чтобы долго не ждать генерацию
#]


# Метод apply_chat_template() используется для форматирования сообщений в одну строку, формат
# которой ориентирова на использование чат-ориентированных языковых моделей.
text = tokenizer.apply_chat_template(
    message2,
    tokenize=False,
    add_generation_prompt=True
)

# В текст добавляются специальные  токены-метки для указания структуры разговора.
print(text[0:160])

# Токенизатор разбивает текст на токены.
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# input_ids — токены в числовом виде, которые представляют собой уникальный номер (ID) из словаря модели.
# attention_mask — задает маску, которая показывает позиции реальных токенов и дополнений (padding) или служебных токенов.
# Модели обрабатывают батчи текстов одинаковой длины. Чтобы выровнять длину реальных текстов короткие тексты дополняются специальным токеном [PAD].
# Таким токенам в attention_mask соответствуют нули, чтобы модель их не обрабатывала.
print(model_inputs)

# Токенизированный текст подаем в модель.
# max_new_tokens - задает максимальное число генерируемых в ответе токенов.
# Можно указать и другие параметры:
# temperature - регулирует "случайность" в выборе следующего токена (<1 — более предсказуемо, >1 — более хаотично).
# top_k - ограничивает выбор следующего токена K наиболее вероятными
# repetition_penalty - штраф за повторяющиеся токены (>1 - штраф, 1 - без штрафа).
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
#    temperature=0.9,
#    top_k=50,
#    repetition_penalty=1.2
)

# Так как модель возвращает и промпт и сгенерированные токены, выделяем только ответ.
generated_ids_ = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Преобразуем ID токенов обратно в слова.
response = tokenizer.batch_decode(generated_ids_, skip_special_tokens=True)[0]

# Смотрим что получилось.
print(response)

##########################################33333333333333333333333#########################################################
message3 = [
    {"role": "system", "content": "Кто предложил цепное правило дифференцирования и в каком году? Ответ давай на русском языке."},
    {"role": "user", "content": data[0:8291]} # берем всё
]

# Для YandexGPT инструкции пишутся от имени пользователя
#messages = [
#    {"role": "user", "content": "Суммаризируй текст. Определи жанр текста, выдели основную информацию. Ответ давай на русском языке."+data[0:5000]} # берем только часть чтобы долго не ждать генерацию
#]


# Метод apply_chat_template() используется для форматирования сообщений в одну строку, формат
# которой ориентирова на использование чат-ориентированных языковых моделей.
text = tokenizer.apply_chat_template(
    message3,
    tokenize=False,
    add_generation_prompt=True
)

# В текст добавляются специальные  токены-метки для указания структуры разговора.
print(text[0:160])

# Токенизатор разбивает текст на токены.
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# input_ids — токены в числовом виде, которые представляют собой уникальный номер (ID) из словаря модели.
# attention_mask — задает маску, которая показывает позиции реальных токенов и дополнений (padding) или служебных токенов.
# Модели обрабатывают батчи текстов одинаковой длины. Чтобы выровнять длину реальных текстов короткие тексты дополняются специальным токеном [PAD].
# Таким токенам в attention_mask соответствуют нули, чтобы модель их не обрабатывала.
print(model_inputs)

# Токенизированный текст подаем в модель.
# max_new_tokens - задает максимальное число генерируемых в ответе токенов.
# Можно указать и другие параметры:
# temperature - регулирует "случайность" в выборе следующего токена (<1 — более предсказуемо, >1 — более хаотично).
# top_k - ограничивает выбор следующего токена K наиболее вероятными
# repetition_penalty - штраф за повторяющиеся токены (>1 - штраф, 1 - без штрафа).
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
#    temperature=0.9,
#    top_k=50,
#    repetition_penalty=1.2
)

# Так как модель возвращает и промпт и сгенерированные токены, выделяем только ответ.
generated_ids_ = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Преобразуем ID токенов обратно в слова.
response = tokenizer.batch_decode(generated_ids_, skip_special_tokens=True)[0]

# Смотрим что получилось.
print(response)