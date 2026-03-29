#!/usr/bin/env python3
"""
BERT模型训练数据集制作脚本
功能：读取datasets文件夹下的所有医疗文献数据，统一格式后输出用于BERT分类训练的数据集
"""

import os
import json
import pandas as pd
import glob
from pathlib import Path
import re

DATASETS_DIR = Path("/home/next_lb/桌面/next/基于自然语言处理的医疗文献智能检索系统的设计与实现/code_and_datasets/datasets")
OUTPUT_DIR = Path("/home/next_lb/桌面/next/基于自然语言处理的医疗文献智能检索系统的设计与实现/code_and_datasets/Data_Preprocessing")

LABEL_MAPPING = {
    "肿瘤学": 0,
    "内分泌代谢": 1,
    "心血管": 2,
    "其他": 3,
    "感染性疾病": 4,
    "中医药": 5,
    "神经内科": 6
}

KEYWORD_TO_LABEL = {
    "肿瘤": "肿瘤学",
    "癌症": "肿瘤学",
    "癌": "肿瘤学",
    "肺癌": "肿瘤学",
    "肝癌": "肿瘤学",
    "乳腺癌": "肿瘤学",
    "胃癌": "肿瘤学",
    "结肠癌": "肿瘤学",
    "直肠癌": "肿瘤学",
    "甲状腺": "肿瘤学",
    "前列腺": "肿瘤学",
    "胰腺": "肿瘤学",
    "白血病": "肿瘤学",
    "淋巴瘤": "肿瘤学",
    "骨髓瘤": "肿瘤学",
    "宫颈癌": "肿瘤学",
    "卵巢癌": "肿瘤学",
    "子宫内膜": "肿瘤学",
    
    "糖尿病": "内分泌代谢",
    "甲状腺功能": "内分泌代谢",
    "甲亢": "内分泌代谢",
    "甲减": "内分泌代谢",
    "骨质疏松": "内分泌代谢",
    "肥胖": "内分泌代谢",
    "代谢": "内分泌代谢",
    "皮质醇": "内分泌代谢",
    "肾上腺": "内分泌代谢",
    "垂体": "内分泌代谢",
    "胰岛素": "内分泌代谢",
    
    "高血压": "心血管",
    "冠心病": "心血管",
    "心肌": "心血管",
    "心律": "心血管",
    "心力衰竭": "心血管",
    "动脉粥样": "心血管",
    "血栓": "心血管",
    "脑卒中": "心血管",
    "中风": "心血管",
    "心绞痛": "心血管",
    "心肌梗死": "心血管",
    "心脏": "心血管",
    "血管": "心血管",
    "血压": "心血管",
    
    "感染": "感染性疾病",
    "细菌": "感染性疾病",
    "病毒": "感染性疾病",
    "真菌": "感染性疾病",
    "肺炎": "感染性疾病",
    "肝炎": "感染性疾病",
    "结核": "感染性疾病",
    "HIV": "感染性疾病",
    "艾滋病": "感染性疾病",
    "流感": "感染性疾病",
    "新冠": "感染性疾病",
    "败血症": "感染性疾病",
    "脑膜炎": "感染性疾病",
    
    "中医": "中医药",
    "中药": "中医药",
    "针灸": "中医药",
    "推拿": "中医药",
    "气功": "中医药",
    "经络": "中医药",
    "穴位": "中医药",
    "辨证": "中医药",
    "方剂": "中医药",
    "草药": "中医药",
    
    "神经": "神经内科",
    "阿尔茨海默": "神经内科",
    "帕金森": "神经内科",
    "癫痫": "神经内科",
    "脑炎": "神经内科",
    "脑梗": "神经内科",
    "脊髓": "神经内科",
    "周围神经": "神经内科",
    "头痛": "神经内科",
    "眩晕": "神经内科",
    "失眠": "神经内科",
    "抑郁症": "神经内科",
    "焦虑": "神经内科",
    "精神": "神经内科",
    "认知": "神经内科",
    "痴呆": "神经内科",
}


def load_dataset_2():
    """加载数据集_2 (已有topic标签)"""
    print("加载数据集_2...")
    file_path = DATASETS_DIR / "数据集_2" / "medical_literature_dataset.csv"
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    records = []
    for _, row in df.iterrows():
        text = build_text(row)
        label = row.get('topic', '其他')
        if pd.isna(label):
            label = '其他'
        records.append({
            'text': text,
            'label': label,
            'label_id': LABEL_MAPPING.get(label, LABEL_MAPPING['其他']),
            'source': 'dataset_2'
        })
    
    print(f"  数据集_2: {len(records)} 条记录")
    return records


def load_dataset_1():
    """加载数据集_1 (英文键JSON)"""
    print("加载数据集_1...")
    file_path = DATASETS_DIR / "数据集_1" / "medical_literature.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for item in data:
        text = build_text_from_dict(item, 'en')
        label = infer_label_from_text(text)
        records.append({
            'text': text,
            'label': label,
            'label_id': LABEL_MAPPING.get(label, LABEL_MAPPING['其他']),
            'source': 'dataset_1'
        })
    
    print(f"  数据集_1: {len(records)} 条记录")
    return records


def load_dataset_3():
    """加载数据集_3 (中文键JSON)"""
    print("加载数据集_3...")
    file_path = DATASETS_DIR / "数据集_3" / "medical_literature.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for item in data:
        text = build_text_from_dict(item, 'zh')
        label = infer_label_from_text(text)
        records.append({
            'text': text,
            'label': label,
            'label_id': LABEL_MAPPING.get(label, LABEL_MAPPING['其他']),
            'source': 'dataset_3'
        })
    
    print(f"  数据集_3: {len(records)} 条记录")
    return records


def load_dataset_4_5():
    """加载数据集_4和5 (医疗文献JSON)"""
    print("加载数据集_4和5...")
    
    records = []
    for dataset_num in ['4', '5']:
        json_files = glob.glob(str(DATASETS_DIR / f"数据集_{dataset_num}" / "医疗文献_*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    text = build_text_from_medical_json(item)
                    label = infer_label_from_text(text)
                    records.append({
                        'text': text,
                        'label': label,
                        'label_id': LABEL_MAPPING.get(label, LABEL_MAPPING['其他']),
                        'source': f'dataset_{dataset_num}'
                    })
            except Exception as e:
                print(f"  警告: 加载 {file_path} 出错: {e}")
    
    print(f"  数据集_4和5: {len(records)} 条记录")
    return records


def load_dataset_6():
    """加载数据集_6 (chinese_medical_articles.csv)"""
    print("加载数据集_6...")
    file_path = DATASETS_DIR / "数据集_6" / "chinese_medical_articles.csv"
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    records = []
    for _, row in df.iterrows():
        text = build_text_from_csv_row(row)
        label = infer_label_from_text(text)
        records.append({
            'text': text,
            'label': label,
            'label_id': LABEL_MAPPING.get(label, LABEL_MAPPING['其他']),
            'source': 'dataset_6'
        })
    
    print(f"  数据集_6: {len(records)} 条记录")
    return records


def load_dataset_7():
    """加载数据集_7 (chinese_medical_from_pubmed.csv)"""
    print("加载数据集_7...")
    file_path = DATASETS_DIR / "数据集_7" / "chinese_medical_from_pubmed.csv"
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    records = []
    for _, row in df.iterrows():
        text = build_text_from_pubmed(row)
        label = infer_label_from_text(text)
        records.append({
            'text': text,
            'label': label,
            'label_id': LABEL_MAPPING.get(label, LABEL_MAPPING['其他']),
            'source': 'dataset_7'
        })
    
    print(f"  数据集_7: {len(records)} 条记录")
    return records


def load_dataset_8():
    """加载数据集_8"""
    print("加载数据集_8...")
    
    records = []
    json_files = glob.glob(str(DATASETS_DIR / "数据集_8" / "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    text = build_text_from_dict(item, 'en')
                    label = infer_label_from_text(text)
                    records.append({
                        'text': text,
                        'label': label,
                        'label_id': LABEL_MAPPING.get(label, LABEL_MAPPING['其他']),
                        'source': 'dataset_8'
                    })
        except Exception as e:
            print(f"  警告: 加载 {file_path} 出错: {e}")
    
    print(f"  数据集_8: {len(records)} 条记录")
    return records


def build_text(row):
    """从DataFrame行构建文本"""
    parts = []
    
    title = row.get('title', '')
    if pd.notna(title) and title:
        parts.append(str(title))
    
    abstract = row.get('abstract', '')
    if pd.notna(abstract) and abstract:
        parts.append(str(abstract))
    
    keywords = row.get('keywords', '')
    if pd.notna(keywords) and keywords:
        parts.append(str(keywords))
    
    return " ".join(parts)


def build_text_from_dict(item, lang):
    """从字典构建文本"""
    parts = []
    
    if lang == 'en':
        title = item.get('title', '')
        abstract = item.get('abstract', '')
        keyword = item.get('search_keyword', '')
    else:
        title = item.get('标题', '')
        abstract = item.get('摘要', '')
        keyword = item.get('搜索关键词', '')
    
    if title:
        parts.append(str(title))
    if abstract:
        parts.append(str(abstract))
    if keyword:
        parts.append(str(keyword))
    
    return " ".join(parts)


def build_text_from_medical_json(item):
    """从医疗文献JSON构建文本"""
    parts = []
    
    title = item.get('title', '')
    if title:
        parts.append(str(title))
    
    abstract = item.get('abstract', '')
    if abstract:
        parts.append(str(abstract))
    
    keywords = item.get('keywords', '')
    if keywords:
        parts.append(str(keywords))
    
    return " ".join(parts)


def build_text_from_csv_row(row):
    """从CSV行构建文本"""
    parts = []
    
    title = row.get('标题', '')
    if pd.notna(title) and title:
        parts.append(str(title))
    
    abstract = row.get('摘要', '')
    if pd.notna(abstract) and abstract:
        parts.append(str(abstract))
    
    keywords = row.get('关键词', '')
    if pd.notna(keywords) and keywords:
        parts.append(str(keywords))
    
    return " ".join(parts)


def build_text_from_pubmed(row):
    """从PubMed数据构建文本"""
    parts = []
    
    for col in ['Title', 'Abstract', 'Keywords', 'MeSH Terms']:
        val = row.get(col, '')
        if pd.notna(val) and val:
            parts.append(str(val))
    
    return " ".join(parts)


def infer_label_from_text(text):
    """根据文本内容推断标签"""
    if not text:
        return '其他'
    
    text_lower = text.lower()
    
    for keyword, label in KEYWORD_TO_LABEL.items():
        if keyword in text_lower or keyword in text:
            return label
    
    return '其他'


def clean_text(text):
    """清洗文本"""
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text[:512]


def main():
    """主函数"""
    print("=" * 60)
    print("开始制作BERT训练数据集")
    print("=" * 60)
    
    all_records = []
    
    all_records.extend(load_dataset_2())
    all_records.extend(load_dataset_1())
    all_records.extend(load_dataset_3())
    all_records.extend(load_dataset_4_5())
    all_records.extend(load_dataset_6())
    all_records.extend(load_dataset_7())
    all_records.extend(load_dataset_8())
    
    print("\n清洗文本数据...")
    for record in all_records:
        record['text'] = clean_text(record['text'])
    
    all_records = [r for r in all_records if r['text']]
    
    print(f"\n总共加载 {len(all_records)} 条记录")
    
    df = pd.DataFrame(all_records)
    
    print("\n标签分布:")
    print(df['label'].value_counts())
    
    output_csv = OUTPUT_DIR / "bert_train_dataset.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n数据集已保存到: {output_csv}")
    
    output_json = OUTPUT_DIR / "bert_train_dataset.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"数据集已保存到: {output_json}")
    
    label_stats = df.groupby(['label', 'label_id']).size().reset_index(name='count')
    label_stats = label_stats.sort_values('label_id')
    print("\n标签映射:")
    for _, row in label_stats.iterrows():
        print(f"  {row['label']}: {row['label_id']} ({row['count']} 条)")
    
    print("\n" + "=" * 60)
    print("数据集制作完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
