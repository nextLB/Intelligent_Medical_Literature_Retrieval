#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer


class BertClassifier:
    def __init__(self, model_path='bert-base-chinese', num_classes=15, dropout=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.max_length = 256
        
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertClassificationModel(model_path, num_classes, dropout)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.label_mapping = {
            0: '肿瘤学',
            1: '内分泌代谢',
            2: '心血管',
            3: '神经内科',
            4: '感染性疾病',
            5: '中医药',
            6: '呼吸系统',
            7: '消化系统',
            8: '泌尿肾脏',
            9: '儿科',
            10: '医疗AI/技术',
            11: '公共卫生',
            12: '风湿免疫',
            13: '精神心理',
            14: '其他'
        }
    
    def predict(self, text):
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
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        return {
            'category_id': pred,
            'category_name': self.label_mapping[pred],
            'confidence': confidence,
            'probabilities': {self.label_mapping[i]: probs[0][i].item() for i in range(self.num_classes)}
        }
    
    def predict_batch(self, texts):
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


class BertClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super(BertClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


bert_classifier = None


def get_bert_classifier():
    global bert_classifier
    if bert_classifier is None:
        bert_classifier = BertClassifier()
    return bert_classifier


if __name__ == '__main__':
    classifier = get_bert_classifier()
    test_text = "本研究探讨了肺癌的靶向治疗方案，分析了PD-1抑制剂在晚期非小细胞肺癌患者中的疗效。"
    result = classifier.predict(test_text)
    print(f"预测结果: {result}")
