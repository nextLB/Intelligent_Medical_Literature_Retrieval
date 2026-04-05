#!/usr/bin/env python3
"""
数据重标注脚本 - 将原有的7分类扩展为15分类
根据关键词模式自动重新标注训练数据
"""

import csv
import os
import json

# 新的分类体系
LABEL_MAPPING = {
    0: ('肿瘤学', ['cancer', 'tumor', 'carcinoma', 'oncolog', 'neoplasm', 'malignan', 'leukemia', 'lymphoma', 'myeloma', 'sarcoma']),
    1: ('内分泌代谢', ['diabetes', 'diabetic', 'glucose', 'insulin', 'thyroid', 'metabolic', 'obesity', 'lipid', 'cholesterol', 'triglyceride', 'parathyroid', 'cushing', 'acromegaly']),
    2: ('心血管', ['hypertension', 'heart', 'cardiac', 'coronary', 'arrhythmia', 'myocardial', 'atrial', 'ventricular', 'aortic', 'aneurysm', 'vascular', 'atheroscler', 'angina', 'heart failure', 'cardiomyopathy']),
    3: ('神经内科', ['stroke', 'alzheimer', 'parkinson', 'dementia', 'cerebrovasc', 'brain ischemia', 'neural', 'neuron', 'epilepsy', 'migraine', 'multiple sclerosis', 'encephal', 'meningitis']),
    4: ('感染性疾病', ['virus', 'viral', 'bacterial', 'infection', 'hiv', 'aids', 'tuberculos', 'covid', 'pandemic', 'parasit', 'sepsis', 'influenza', 'hepatitis b', 'hepatitis c', 'malaria']),
    5: ('中医药', ['traditional chinese', 'chinese medicine', 'acupuncture', 'herb', 'tcm', 'acupoint', 'moxibustion', 'auricular', 'cupping', 'tuina']),
    6: ('呼吸系统', ['copd', 'chronic obstructive', 'pulmonary disease', 'pulmonary', 'asthma', 'bronchial', 'pneumonia', 'respiratory', 'lung', 'pneumoconiosis', 'silicosis', 'airway', 'bronchiectasis', 'tuberculos']),
    7: ('消化系统', ['liver', 'hepatic', 'hepatitis', 'cirrhosis', 'biliary', 'gallbladder', 'pancrea', 'gastric', 'gastrointestinal', 'esophag', 'intestin', 'colon', 'bowel', 'ulcer', 'gastritis', 'crohn', 'colitis']),
    8: ('泌尿肾脏', ['renal', 'kidney', 'nephritis', 'urinary', 'bladder', 'prostate', 'nephrotic', 'dialysis', 'glomerulonephritis']),
    9: ('儿科', ['pediatric', 'child', 'infant', 'neonatal', 'newborn', 'pediatr', 'congenital', 'birth defect']),
    10: ('医疗AI/技术', ['artificial intelligence', 'machine learning', 'deep learning', 'ai', '5g', 'blockchain', 'wearable', 'telemedicine', 'imaging', 'algorithm', 'neural network', 'computer vision', 'nlp', 'natural language']),
    11: ('公共卫生', ['public health', 'epidemiolog', 'epidemic', 'prevention', 'screening', 'healthcare', 'population', 'survey']),
    12: ('风湿免疫', ['rheumat', 'arthritis', 'lupus', 'autoimmune', 'immunolog', 'sle', 'rheumatoid', 'sjögren', 'scleroderma']),
    13: ('精神心理', ['psychiatr', 'depression', 'anxiety', 'mental', 'schizophrenia', 'bipolar', 'psycholog', 'autism', 'adhd', 'personality disorder']),
    14: ('其他', [])  # 兜底类别
}

# 反向映射
LABEL_MAPPING_REVERSE = {cat_id: name for cat_id, (name, _) in LABEL_MAPPING.items()}

# 类别描述
CATEGORY_DESCRIPTIONS = {
    0: '包括肺癌、胃癌、肝癌、乳腺癌等各类肿瘤研究',
    1: '包括糖尿病、甲状腺疾病、骨质疏松、肥胖等',
    2: '包括高血压、冠心病、心力衰竭、心律失常等',
    3: '包括阿尔茨海默病、帕金森病、脑卒中等',
    4: '包括肺炎、肝炎、结核病、COVID-19等',
    5: '包括中药、针灸、推拿等中医药研究',
    6: '包括慢阻肺、哮喘、肺炎、肺纤维化等',
    7: '包括胃炎、肝炎、肝硬化、胰腺炎等',
    8: '包括肾炎、尿路感染、前列腺疾病等',
    9: '包括儿童疾病、新生儿疾病等',
    10: '包括人工智能、机器学习、5G医疗等',
    11: '包括流行病学、疾病预防、健康管理等',
    12: '包括类风湿性关节炎、系统性红斑狼疮等',
    13: '包括抑郁症、焦虑症、精神分裂症等',
    14: '其他医学领域的研究'
}

def classify_text(text):
    """根据关键词模式对文本进行分类"""
    text_lower = text.lower()
    
    for cat_id, (cat_name, keywords) in LABEL_MAPPING.items():
        if not keywords:  # 跳过"其他"
            continue
        for kw in keywords:
            if kw in text_lower:
                return cat_id
    
    return 14  # 其他

def relabel_dataset(input_csv, output_csv):
    """重标注数据集"""
    print(f"读取原始数据: {input_csv}")
    
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"原始数据: {len(rows)} 条")
    
    # 重新标注
    new_rows = []
    label_counts = {i: 0 for i in range(15)}
    
    for row in rows:
        text = row.get('text', '')
        new_label = classify_text(text)
        label_counts[new_label] += 1
        
        new_row = {
            'text': text,
            'label_id': new_label,
            'label_name': LABEL_MAPPING_REVERSE[new_label]
        }
        new_rows.append(new_row)
    
    # 写入新CSV
    print(f"\n写入新数据: {output_csv}")
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['text', 'label_id', 'label_name']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    
    # 打印统计
    print("\n=== 新分类分布 ===")
    total = len(rows)
    for cat_id in sorted(LABEL_MAPPING_REVERSE.keys()):
        count = label_counts[cat_id]
        pct = count / total * 100
        print(f"  {cat_id:2d}. {LABEL_MAPPING_REVERSE[cat_id]:<12s}: {count:5d} 条 ({pct:5.1f}%)")
    
    print(f"\n总计: {len(rows)} 条")
    print("\n数据重标注完成!")
    
    return label_counts

def update_system_files():
    """更新系统文件中的标签映射"""
    system_code = "/home/next_lb/桌面/next/基于自然语言处理的医疗文献智能检索系统的设计与实现/Intelligent_Medical_Literature_Retrieval/system_code"
    
    # 更新views.py中的标签映射
    views_path = os.path.join(system_code, "literature/views.py")
    
    new_mapping = "LABEL_MAPPING_REVERSE = {\n"
    for cat_id in sorted(LABEL_MAPPING_REVERSE.keys()):
        new_mapping += f"    {cat_id}: '{LABEL_MAPPING_REVERSE[cat_id]}',"
    new_mapping += "\n}"
    
    print(f"\n需要在 {views_path} 中更新 LABEL_MAPPING_REVERSE")
    print("请手动更新以下内容:")
    print(new_mapping)
    
    # 更新ml_models中的标签映射
    for model_name in ['bert_classifier.py', 'text_vectorizer.py']:
        model_path = os.path.join(system_code, "ml_models", model_name)
        if os.path.exists(model_path):
            print(f"\n需要在 {model_path} 中更新标签映射")

if __name__ == '__main__':
    base_dir = "/home/next_lb/桌面/next/基于自然语言处理的医疗文献智能检索系统的设计与实现/Intelligent_Medical_Literature_Retrieval"
    data_dir = os.path.join(base_dir, "Data_Preprocessing")
    
    input_csv = os.path.join(data_dir, "bert_train_dataset.csv")
    output_csv = os.path.join(data_dir, "bert_train_dataset_new.csv")
    
    # 执行重标注
    label_counts = relabel_dataset(input_csv, output_csv)
    
    # 输出需要更新的文件信息
    update_system_files()
    
    print("\n" + "="*60)
    print("下一步操作:")
    print("="*60)
    print("1. 移动新的训练数据: mv bert_train_dataset_new.csv bert_train_dataset.csv")
    print("2. 更新 views.py 中的 LABEL_MAPPING_REVERSE")
    print("3. 更新 ml_models/bert_classifier.py 中的标签映射")
    print("4. 更新 ml_models/text_vectorizer.py (如需要)")
    print("5. 运行数据库迁移更新分类")
    print("6. 重新训练模型")
