#!/usr/bin/env python3
"""
BERT模型训练脚本
用于训练中文医学文献分类模型
适配12GB RTX 3060显卡
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


CONFIG = {
    'model_name': 'bert-base-chinese',
    'max_length': 256,
    'batch_size': 16,
    'num_epochs': 5,
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'dropout': 0.3,
    'num_classes': 7,
    'seed': 42,
    'data_dir': '../Data_Preprocessing',
    'output_dir': './checkpoints',
    'log_dir': './logs'
}

LABEL_MAPPING = {
    '肿瘤学': 0,
    '内分泌代谢': 1,
    '心血管': 2,
    '其他': 3,
    '感染性疾病': 4,
    '中医药': 5,
    '神经内科': 6
}

LABEL_MAPPING_REVERSE = {v: k for k, v in LABEL_MAPPING.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class MedicalTestDataset(Dataset):
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


def load_data(data_dir):
    csv_path = os.path.join(data_dir, 'bert_train_dataset.csv')
    json_path = os.path.join(data_dir, 'bert_train_dataset.json')
    
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    elif os.path.exists(json_path):
        print(f"Loading data from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise FileNotFoundError(f"Dataset not found in {data_dir}")
    
    texts = df['text'].tolist()
    labels = df['label_id'].tolist()
    
    return texts, labels


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def train():
    print("=" * 60)
    print("BERT Medical Text Classification Training")
    print("=" * 60)
    
    set_seed(CONFIG['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    
    print("Loading data...")
    texts, labels = load_data(CONFIG['data_dir'])
    print(f"Total samples: {len(texts)}")
    
    label_counts = pd.Series(labels).value_counts().sort_index()
    print("\nLabel distribution:")
    for label_id, count in label_counts.items():
        print(f"  {LABEL_MAPPING_REVERSE[label_id]}: {count}")
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, 
        test_size=0.15, 
        random_state=CONFIG['seed'],
        stratify=labels
    )
    
    print(f"\nTrain samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    train_dataset = MedicalDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'])
    val_dataset = MedicalDataset(val_texts, val_labels, tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print("\nLoading model...")
    from model import BertClassifier
    model = BertClassifier(
        num_classes=CONFIG['num_classes'],
        model_name=CONFIG['model_name'],
        dropout=CONFIG['dropout']
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    
    total_steps = len(train_loader) * CONFIG['num_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\n" + "=" * 60)
    print("Start Training")
    print("=" * 60)
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, os.path.join(CONFIG['output_dir'], 'best_model.pt'))
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': CONFIG
        }, os.path.join(CONFIG['output_dir'], f'checkpoint_epoch_{epoch+1}.pt'))
    
    with open(os.path.join(CONFIG['log_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Best Val Acc: {best_val_acc:.4f}")
    print("=" * 60)
    
    return model, history


def predict(model, texts, tokenizer, device):
    model.eval()
    
    dataset = MedicalTestDataset(texts, tokenizer, CONFIG['max_length'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
    
    return predictions


if __name__ == '__main__':
    train()
