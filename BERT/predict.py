#!/usr/bin/env python3
"""
模型预测脚本
用于对新的医学文献进行分类预测
"""

import os
import json
import torch
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model import BertClassifier
from config import CONFIG, LABEL_MAPPING_REVERSE


class MedicalTestDataset:
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def load_model(checkpoint_path, device):
    model = BertClassifier(
        num_classes=CONFIG['num_classes'],
        model_name=CONFIG['model_name'],
        dropout=CONFIG['dropout']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def predict(model, texts, tokenizer, device, batch_size=16):
    model.eval()
    
    dataset = MedicalTestDataset(texts, tokenizer, CONFIG['max_length'])
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return predictions, probabilities


def predict_from_file(model_path, input_file, output_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    model = load_model(model_path, device)
    
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        texts = df['text'].tolist()
    elif input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [item['text'] for item in data]
    else:
        raise ValueError("Input file must be .csv or .json")
    
    predictions, probabilities = predict(model, texts, tokenizer, device)
    
    results = []
    for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
        results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'predicted_label': LABEL_MAPPING_REVERSE[pred],
            'label_id': pred,
            'confidence': float(max(prob))
        })
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    
    return results


def predict_single(model_path, text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    model = load_model(model_path, device)
    
    predictions, probabilities = predict(model, [text], tokenizer, device)
    
    pred = predictions[0]
    prob = probabilities[0]
    
    print(f"\nText: {text[:100]}...")
    print(f"Predicted Label: {LABEL_MAPPING_REVERSE[pred]}")
    print(f"Confidence: {max(prob):.4f}")
    print("\nAll probabilities:")
    for i, p in enumerate(prob):
        print(f"  {LABEL_MAPPING_REVERSE[i]}: {p:.4f}")
    
    return LABEL_MAPPING_REVERSE[pred], max(prob)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py <checkpoint_path> <text>")
        print("  python predict.py <checkpoint_path> --file <input_file> [--output <output_file>]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    if len(sys.argv) > 3 and sys.argv[2] == '--file':
        input_file = sys.argv[3]
        output_file = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == '--output' else None
        predict_from_file(checkpoint_path, input_file, output_file)
    else:
        text = ' '.join(sys.argv[2:])
        predict_single(checkpoint_path, text)
