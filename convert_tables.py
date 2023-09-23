import sys
import torch
import pandas as pd
import numpy as np
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineSimilarity

A = pd.read_csv(sys.argv[1])
B = pd.read_csv(sys.argv[2])
template = pd.read_csv(sys.argv[3])

A_for_comparison = A.iloc[:8]
B_for_comparison = B.iloc[:8]
template = template.iloc[:8]

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModel.from_pretrained("dslim/bert-base-NER")
# model.cuda()  # uncomment it if you have a GPU

# Дефолтная функция, шла в комплекте с моделью rubert
def embed_bert(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=1024) # Модель сама создаёт пэддинги и маску.
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

A_encoded_cols = {}
for column in A_for_comparison.columns:
    A_encoded_cols[column] = embed_bert(A_for_comparison[column].astype(str).tolist(), model, tokenizer)

B_encoded_cols = {}
for column in B_for_comparison.columns:
    B_encoded_cols[column] = embed_bert(B_for_comparison[column].astype(str).tolist(), model, tokenizer)

T_encoded_cols = {}
for column in template.columns:
    T_encoded_cols[column] = embed_bert(template[column].astype(str).tolist(), model, tokenizer)

cos_simsA = {}
for t_col_name, t_col_encoded in T_encoded_cols.items():
    for a_col_name, a_col_encoded in A_encoded_cols.items():
        cos_simsA[(t_col_name, a_col_name)] = CosineSimilarity(dim=0)(torch.from_numpy(t_col_encoded), torch.from_numpy(a_col_encoded)).mean()

# Создаем список с именами столбцов, для которых ищем наиболее схожие
columns = template.columns

# Создаем пустой список для хранения результатов ПО ТАБЛИЦЕ А
resultsA = []

# Для каждого столбца из списка
for column in columns:
  # Инициализируем максимальное сходство нулем
  max_similarity = 0
  # Инициализируем наиболее схожий столбец пустой строкой
  most_similar = ""
  # Для каждого ключа и значения из словаря с данными
  for key, value in cos_simsA.items():
    # Если первый элемент ключа совпадает с текущим столбцом
    if key[0] == column:
      # Если значение больше максимального сходства
      if value > max_similarity:
        # Обновляем максимальное сходство и наиболее схожий столбец
        max_similarity = value
        most_similar = key[1]
  # Добавляем в список результатов пару текущий столбец - наиболее схожий столбец
  resultsA.append((column, most_similar))

cos_simsB = {}
for t_col_name, t_col_encoded in T_encoded_cols.items():
    for b_col_name, b_col_encoded in B_encoded_cols.items():
        cos_simsB[(t_col_name, b_col_name)] = CosineSimilarity(dim=0)(torch.from_numpy(t_col_encoded), torch.from_numpy(b_col_encoded)).mean()


# Создаем пустой список для хранения результатов ПО ТАБЛИЦЕ Б
resultsB = []

# Для каждого столбца из списка
for column in columns:
  # Инициализируем максимальное сходство нулем
  max_similarity = 0
  # Инициализируем наиболее схожий столбец пустой строкой
  most_similar = ""
  # Для каждого ключа и значения из словаря с данными
  for key, value in cos_simsB.items():
    # Если первый элемент ключа совпадает с текущим столбцом
    if key[0] == column:
      # Если значение больше максимального сходства
      if value > max_similarity:
        # Обновляем максимальное сходство и наиболее схожий столбец
        max_similarity = value
        most_similar = key[1]
  # Добавляем в список результатов пару текущий столбец - наиболее схожий столбец
  resultsB.append((column, most_similar))

A_table_final = pd.DataFrame(columns=template.columns)
for tpl in resultsA:
    A_table_final[tpl[0]] = A[tpl[1]]

B_table_final = pd.DataFrame(columns=template.columns)

for tpl in resultsB:
    B_table_final[tpl[0]] = B[tpl[1]]
  
def convert_dateA(date_str):
    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
    return date_obj.strftime('%m.%d.%y')

def convert_dateB(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return date_obj.strftime('%m.%d.%y')


def convert_policy_number(policy_number_str):
    policy_number_str = policy_number_str.replace('-', '')
    policy_number_str = re.sub('[^0-9a-zA-Z]+', '', policy_number_str)
    return policy_number_str.upper()

A_table_final['Date'] = A_table_final['Date'].apply(convert_dateA)
B_table_final['Date'] = B_table_final['Date'].apply(convert_dateB)
A_table_final['PolicyNumber'] = A_table_final['PolicyNumber'].apply(convert_policy_number)
B_table_final['PolicyNumber'] = B_table_final['PolicyNumber'].apply(convert_policy_number)

final_table = pd.concat([A_table_final, B_table_final], axis=0, ignore_index=True)
final_table.to_csv('result.csv')
print(final_table)