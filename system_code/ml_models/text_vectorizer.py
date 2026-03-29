#!/usr/bin/env python3
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import pickle
import os


class TextVectorizer:
    def __init__(self, model_name='bert-base-chinese'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_length = 256
    
    def get_vector(self, text):
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return vector
    
    def get_vectors(self, texts):
        vectors = []
        for text in texts:
            vectors.append(self.get_vector(text))
        return np.array(vectors)
    
    def text_to_vector(self, text):
        return self.get_vector(text)
    
    def save_vectors(self, vectors, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(vectors, f)
    
    def load_vectors(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


text_vectorizer = None


def get_text_vectorizer():
    global text_vectorizer
    if text_vectorizer is None:
        text_vectorizer = TextVectorizer()
    return text_vectorizer


if __name__ == '__main__':
    vectorizer = get_text_vectorizer()
    test_text = "这是一段医学文献的摘要，讨论了糖尿病的治疗方法。"
    vector = vectorizer.get_vector(test_text)
    print(f"向量维度: {vector.shape}")
    print(f"向量前5个值: {vector[:5]}")
