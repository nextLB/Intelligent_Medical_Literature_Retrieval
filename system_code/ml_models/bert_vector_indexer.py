#!/usr/bin/env python3
import os
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import pickle
import time


class BertVectorIndexer:
    def __init__(self, model_name='bert-base-chinese', vector_dim=768):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_length = 256
        
        self.index_vectors = None
        self.index_ids = []
        self.index_titles = []
        self.index_abstracts = []
        self.index_keywords = []
        self.index_authors = []
        self.index_journals = []
        self.index_years = []
        self.index_pdf_urls = []
        self.index_categories = []
        self.index_sources = []
        self.index_file = None
        
    def get_text_vector(self, text):
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
    
    def get_batch_vectors(self, texts, batch_size=32):
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
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
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.extend(batch_vectors)
        
        return np.array(vectors)
    
    def build_index(self, literature_data):
        texts = [f"{lit['title']} {lit['abstract']}" for lit in literature_data]
        
        vectors = self.get_batch_vectors(texts)
        
        self.index_vectors = vectors
        self.index_ids = [lit['id'] for lit in literature_data]
        self.index_titles = [lit['title'] for lit in literature_data]
        self.index_abstracts = [lit.get('abstract', '') for lit in literature_data]
        self.index_keywords = [lit.get('keywords', '') for lit in literature_data]
        self.index_authors = [lit.get('authors', '') for lit in literature_data]
        self.index_journals = [lit.get('journal', '') for lit in literature_data]
        self.index_years = [lit.get('publish_year', 0) for lit in literature_data]
        self.index_pdf_urls = [lit.get('pdf_url', '') for lit in literature_data]
        self.index_categories = [lit.get('category', '') for lit in literature_data]
        self.index_sources = [lit.get('source', 'csv') for lit in literature_data]
        
        return len(vectors)
    
    def add_to_index(self, literature_id, title, abstract, metadata=None):
        text = f"{title} {abstract}"
        vector = self.get_text_vector(text)
        
        if self.index_vectors is None:
            self.index_vectors = np.array([vector])
        else:
            self.index_vectors = np.vstack([self.index_vectors, vector])
        
        self.index_ids.append(literature_id)
        self.index_titles.append(title)
        self.index_abstracts.append(abstract or '')
        self.index_keywords.append(metadata.get('keywords', '') if metadata else '')
        self.index_authors.append(metadata.get('authors', '') if metadata else '')
        self.index_journals.append(metadata.get('journal', '') if metadata else '')
        self.index_years.append(metadata.get('publish_year', 0) if metadata else 0)
        self.index_pdf_urls.append(metadata.get('pdf_url', '') if metadata else '')
        self.index_categories.append(metadata.get('category', '') if metadata else '')
        self.index_sources.append(metadata.get('source', 'csv') if metadata else 'csv')
        
        return vector
    
    def remove_from_index(self, literature_id):
        if literature_id not in self.index_ids:
            return False
        
        idx = self.index_ids.index(literature_id)
        self.index_vectors = np.delete(self.index_vectors, idx, axis=0)
        self.index_ids.pop(idx)
        self.index_titles.pop(idx)
        self.index_abstracts.pop(idx)
        self.index_keywords.pop(idx)
        self.index_authors.pop(idx)
        self.index_journals.pop(idx)
        self.index_years.pop(idx)
        self.index_pdf_urls.pop(idx)
        self.index_categories.pop(idx)
        self.index_sources.pop(idx)
        
        return True
    
    def compute_similarity_cosine(self, query_vector, top_k=10):
        if self.index_vectors is None or len(self.index_vectors) == 0:
            return []
        
        query_vec = query_vector.reshape(1, -1)
        index_vecs = self.index_vectors
        
        query_norm = np.linalg.norm(query_vec)
        index_norms = np.linalg.norm(index_vecs, axis=1)
        
        dot_products = np.dot(query_vec, index_vecs.T)[0]
        
        norms_product = query_norm * index_norms
        norms_product[norms_product == 0] = 1e-10
        
        similarities = dot_products / norms_product
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'index': int(idx),
                    'literature_id': self.index_ids[idx],
                    'title': self.index_titles[idx],
                    'abstract': self.index_abstracts[idx],
                    'similarity': float(similarities[idx]),
                    'keywords': self.index_keywords[idx] if idx < len(self.index_keywords) else '',
                    'authors': self.index_authors[idx] if idx < len(self.index_authors) else '',
                    'journal': self.index_journals[idx] if idx < len(self.index_journals) else '',
                    'publish_year': self.index_years[idx] if idx < len(self.index_years) else 0,
                    'pdf_url': self.index_pdf_urls[idx] if idx < len(self.index_pdf_urls) else '',
                    'category': self.index_categories[idx] if idx < len(self.index_categories) else '',
                    'source': self.index_sources[idx] if idx < len(self.index_sources) else 'csv'
                })
        
        return results
    
    def find_similar_by_id(self, literature_id, top_k=10):
        if literature_id not in self.index_ids:
            return []
        
        idx = self.index_ids.index(literature_id)
        query_vector = self.index_vectors[idx]
        
        all_similarities = []
        for i, vec in enumerate(self.index_vectors):
            if i == idx:
                continue
            
            query_norm = np.linalg.norm(query_vector)
            vec_norm = np.linalg.norm(vec)
            
            if query_norm == 0 or vec_norm == 0:
                continue
            
            similarity = np.dot(query_vector, vec) / (query_norm * vec_norm)
            all_similarities.append({
                'index': i,
                'literature_id': self.index_ids[i],
                'title': self.index_titles[i],
                'abstract': self.index_abstracts[i],
                'similarity': float(similarity),
                'keywords': self.index_keywords[i] if i < len(self.index_keywords) else '',
                'authors': self.index_authors[i] if i < len(self.index_authors) else '',
                'journal': self.index_journals[i] if i < len(self.index_journals) else '',
                'publish_year': self.index_years[i] if i < len(self.index_years) else 0,
                'pdf_url': self.index_pdf_urls[i] if i < len(self.index_pdf_urls) else '',
                'category': self.index_categories[i] if i < len(self.index_categories) else '',
                'source': self.index_sources[i] if i < len(self.index_sources) else 'csv'
            })
        
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return all_similarities[:top_k]
    
    def search_by_text(self, query_text, top_k=10):
        query_vector = self.get_text_vector(query_text)
        return self.compute_similarity_cosine(query_vector, top_k)
    
    def get_index_size(self):
        return len(self.index_ids) if self.index_ids else 0
    
    def save_index(self, filepath):
        data = {
            'vectors': self.index_vectors,
            'ids': self.index_ids,
            'titles': self.index_titles,
            'abstracts': self.index_abstracts,
            'keywords': self.index_keywords,
            'authors': self.index_authors,
            'journals': self.index_journals,
            'years': self.index_years,
            'pdf_urls': self.index_pdf_urls,
            'categories': self.index_categories,
            'sources': self.index_sources,
            'model_name': self.model_name,
            'vector_dim': self.vector_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return True
    
    def load_index(self, filepath):
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.index_vectors = data['vectors']
        self.index_ids = data['ids']
        self.index_titles = data['titles']
        self.index_abstracts = data['abstracts']
        self.index_keywords = data.get('keywords', [])
        self.index_authors = data.get('authors', [])
        self.index_journals = data.get('journals', [])
        self.index_years = data.get('years', [])
        self.index_pdf_urls = data.get('pdf_urls', [])
        self.index_categories = data.get('categories', [])
        self.index_sources = data.get('sources', [])
        
        return True
    
    def clear_index(self):
        self.index_vectors = None
        self.index_ids = []
        self.index_titles = []
        self.index_abstracts = []
        self.index_keywords = []
        self.index_authors = []
        self.index_journals = []
        self.index_years = []
        self.index_pdf_urls = []
        self.index_categories = []
        self.index_sources = []


bert_vector_indexer = None


def get_bert_vector_indexer():
    global bert_vector_indexer
    if bert_vector_indexer is None:
        bert_vector_indexer = BertVectorIndexer()
    return bert_vector_indexer


if __name__ == '__main__':
    indexer = get_bert_vector_indexer()
    
    test_data = [
        {'id': 1, 'title': '肺癌靶向治疗研究', 'abstract': '本文探讨了EGFR突变阳性非小细胞肺癌的靶向治疗方案。'},
        {'id': 2, 'title': '糖尿病药物治疗', 'abstract': '本研究评估了二甲双胍在2型糖尿病患者中的降糖效果。'},
        {'id': 3, 'title': '心血管疾病预防', 'abstract': '本文讨论了高血压和冠心病的预防策略。'},
    ]
    
    indexer.build_index(test_data)
    print(f"索引构建完成，共 {indexer.get_index_size()} 条文献")
    
    results = indexer.search_by_text("肺癌治疗", top_k=3)
    print("\n搜索结果:")
    for r in results:
        print(f"  标题: {r['title']}")
        print(f"  相似度: {r['similarity']:.4f}")
        print()