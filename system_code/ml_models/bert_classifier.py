#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer


CATEGORY_KEYWORDS = {
    '肿瘤学': ['肿瘤', '癌症', '肺癌', '胃癌', '肝癌', '乳腺癌', '结肠癌', '直肠癌', '化疗', '放疗', '靶向', 'PD-1', '免疫治疗', '恶性肿瘤', '腺癌', '鳞癌', ' metastases', 'carcinoma', 'oncology'],
    '内分泌代谢': ['糖尿病', '甲状腺', '代谢', '胰岛素', '血糖', '甲亢', '甲减', '肥胖', '骨质疏松', '内分泌', '垂体', '肾上腺', 'diabetes', 'thyroid', 'metabolism'],
    '心血管': ['心脏', '高血压', '冠心病', '心衰', '心律', '心肌', '血管', '动脉', '血栓', '心血管', '房颤', '心梗', 'cardiac', 'cardiovascular', 'hypertension'],
    '神经内科': ['脑', '神经', '阿尔茨海默', '帕金森', '癫痫', '卒中', '脑卒中', '痴呆', '抑郁', '焦虑', '精神科', '神经内科', 'neurology', 'alzheimer', 'parkinson'],
    '感染性疾病': ['感染', '病毒', '细菌', '肺炎', '结核', '肝炎', 'HIV', 'COVID', '新冠', '传染病', '抗菌', '抗生素', 'infection', 'virus', 'bacteria', 'TB'],
    '中医药': ['中医', '中药', '针灸', '推拿', '穴位', '辨证', '方剂', '经方', '气血', '阴阳', '湿热', 'TCM', 'herbal', 'acupuncture'],
    '呼吸系统': ['肺', '呼吸', '哮喘', '慢阻肺', '肺炎', '支气管', '肺气肿', '呼吸系统', '咳嗽', '气喘', 'respiratory', 'pulmonary', 'asthma', 'COPD'],
    '消化系统': ['胃', '肠', '肝', '胰腺', '消化', '胃炎', '肠炎', '肝炎', '肝硬化', '胰腺炎', '胃肠道', '消化系统', 'gastrointestinal', 'digestive', 'liver'],
    '泌尿肾脏': ['肾', '泌尿', '肾炎', '尿路', '前列腺', '透析', '肾衰', '尿毒症', '膀胱', '泌尿系统', 'nephrology', 'kidney', 'urinary'],
    '儿科': ['儿童', '小儿', '新生儿', '婴儿', '儿科', '先天', '发育', '小儿科', 'pediatric', 'neonatal', 'child'],
    '医疗AI/技术': ['人工智能', 'AI', '机器学习', '深度学习', '算法', '大数据', '互联网', '5G', '远程医疗', '智能', '神经网络', 'artificial intelligence', 'machine learning', 'deep learning'],
    '公共卫生': ['公共卫生', '流行病', '预防', '健康', '疫苗', '防控', '监测', '流行病学', '卫生', 'public health', 'epidemiology', 'prevention'],
    '风湿免疫': ['风湿', '免疫', '类风湿', '红斑狼疮', '系统性', '自身免疫', '关节炎', '干燥综合征', 'rheumatology', 'autoimmune', 'lupus', 'arthritis'],
    '精神心理': ['精神', '心理', '抑郁', '焦虑', '睡眠', '失眠', '精神分裂', '双相', '躁郁', '心理治疗', 'psychiatry', 'psychological', 'depression', 'anxiety'],
    '其他': []
}


def fallback_classify_by_keywords(text):
    text_lower = text.lower()
    scores = {}
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == '其他':
            continue
        score = 0
        for kw in keywords:
            if kw.lower() in text_lower:
                score += 1
        if score > 0:
            scores[category] = score
    
    if not scores:
        return {
            'category_id': 14,
            'category_name': '其他',
            'confidence': 0.5,
            'probabilities': {k: 0.5/15 for k in CATEGORY_KEYWORDS.keys()},
            'method': 'keyword'
        }
    
    max_score = max(scores.values())
    categories = [c for c, s in scores.items() if s == max_score]
    top_category = categories[0]
    
    total_score = sum(scores.values())
    probs = {}
    for cat in CATEGORY_KEYWORDS.keys():
        if cat == '其他':
            probs[cat] = 0.1
        elif cat in scores:
            probs[cat] = (scores[cat] / total_score) * 0.9
        else:
            probs[cat] = 0.1 / 15
    
    cat_id_map = {
        '肿瘤学': 0, '内分泌代谢': 1, '心血管': 2, '神经内科': 3,
        '感染性疾病': 4, '中医药': 5, '呼吸系统': 6, '消化系统': 7,
        '泌尿肾脏': 8, '儿科': 9, '医疗AI/技术': 10, '公共卫生': 11,
        '风湿免疫': 12, '精神心理': 13, '其他': 14
    }
    
    return {
        'category_id': cat_id_map.get(top_category, 14),
        'category_name': top_category,
        'confidence': 0.7,
        'probabilities': probs,
        'method': 'keyword'
    }


class BertClassifier:
    def __init__(self, model_path='bert-base-chinese', num_classes=15, dropout=0.3, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.max_length = 256
        self.model_loaded = False
        
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertClassificationModel(model_path, num_classes, dropout)
        self.model = self.model.to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model_loaded = True
                print(f"成功加载训练好的模型: {checkpoint_path}")
            except Exception as e:
                print(f"加载模型失败，使用基础BERT: {e}")
        else:
            print("未找到训练好的模型，使用基础BERT进行零样本分类")
        
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
        if not self.model_loaded:
            print("使用关键词匹配方法进行分类")
            return fallback_classify_by_keywords(text)
        
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
            'probabilities': {self.label_mapping[i]: probs[0][i].item() for i in range(self.num_classes)},
            'method': 'bert'
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
        checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(checkpoint_dir, '..', 'checkpoints', 'best_model.pt')
        bert_classifier = BertClassifier(checkpoint_path=checkpoint_path)
    return bert_classifier


if __name__ == '__main__':
    classifier = get_bert_classifier()
    test_text = "本研究探讨了肺癌的靶向治疗方案，分析了PD-1抑制剂在晚期非小细胞肺癌患者中的疗效。"
    result = classifier.predict(test_text)
    print(f"预测结果: {result}")
