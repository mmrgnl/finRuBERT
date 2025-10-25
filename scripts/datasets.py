import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

if not os.path.exists('data/sentiment_data'):
    os.makedirs('data/sentiment_data')

parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('--data_path', type=str, help='Path to the text file.')

args = parser.parse_args()

# Чтение данных с поддержкой UTF-8 для русских текстов
data = pd.read_csv(args.data_path, sep='.@', names=['text','label'], encoding='utf-8')

# Добавляем колонку id для совместимости с FinBERT
data['id'] = range(len(data))
data = data[['id', 'text', 'label']]  # Порядок: id, text, label

train, test = train_test_split(data, test_size=0.2, random_state=0)
train, valid = train_test_split(train, test_size=0.1, random_state=0)

# Сохранение с UTF-8 кодировкой и без индексов
train.to_csv('data/sentiment_data/train.csv', sep='\t', index=False, encoding='utf-8')
test.to_csv('data/sentiment_data/test.csv', sep='\t', index=False, encoding='utf-8')
valid.to_csv('data/sentiment_data/validation.csv', sep='\t', index=False, encoding='utf-8')

print(f"Создано файлов:")
print(f"- train.csv: {len(train)} примеров")
print(f"- validation.csv: {len(valid)} примеров") 
print(f"- test.csv: {len(test)} примеров")
print(f"Общее количество: {len(data)} примеров")