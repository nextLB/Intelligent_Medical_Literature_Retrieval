#!/usr/bin/env python3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .text_vectorizer import get_text_vectorizer


class SimilaritySearch:
    def __init__(self):
        self.vectorizer = get_text_vectorizer()
        self.literature_vectors = {}
        self.literature_texts = {}
    
    def add_literature(self, literature_id, title, abstract):
        text = f"{title} {abstract}"
        vector = self.vectorizer.get_vector(text)
        self.literature_vectors[literature_id] = vector
        self.literature_texts[literature_id] = text
        return vector
    
    def remove_literature(self, literature_id):
        if literature_id in self.literature_vectors:
            del self.literature_vectors[literature_id]
            del self.literature_texts[literature_id]
    
    def search_similar(self, query, top_k=10, literature_ids=None):
        query_vector = self.vectorizer.get_vector(query)
        
        candidate_ids = literature_ids if literature_ids else list(self.literature_vectors.keys())
        
        if not candidate_ids:
            return []
        
        vectors = []
        for lid in candidate_ids:
            if lid in self.literature_vectors:
                vectors.append(self.literature_vectors[lid])
            else:
                vectors.append(self.vectorizer.get_vector(self.literature_texts.get(lid, "")))
        
        vectors = np.array(vectors)
        similarities = cosine_similarity([query_vector], vectors)[0]
        
        results = []
        for i, lid in enumerate(candidate_ids):
            results.append({
                'literature_id': lid,
                'similarity': float(similarities[i])
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def find_similar_literatures(self, literature_id, top_k=5):
        if literature_id not in self.literature_vectors:
            return []
        
        source_vector = self.literature_vectors[literature_id]
        other_ids = [lid for lid in self.literature_vectors.keys() if lid != literature_id]
        
        if not other_ids:
            return []
        
        vectors = [self.literature_vectors[lid] for lid in other_ids]
        vectors = np.array(vectors)
        similarities = cosine_similarity([source_vector], vectors)[0]
        
        results = []
        for i, lid in enumerate(other_ids):
            results.append({
                'literature_id': lid,
                'similarity': float(similarities[i])
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def clear_all(self):
        self.literature_vectors.clear()
        self.literature_texts.clear()


similarity_search = None


def get_similarity_search():
    global similarity_search
    if similarity_search is None:
        similarity_search = SimilaritySearch()
    return similarity_search


if __name__ == '__main__':
    search = get_similarity_search()
    search.add_literature(1, "肺癌靶向治疗研究", "本文探讨了EGFR突变阳性非小细胞肺癌的靶向治疗方案...")
    search.add_literature(2, "糖尿病药物治疗", "本研究评估了二甲双胍在2型糖尿病患者中的降糖效果...")
    search.add_literature(3, "心血管疾病预防", "本文讨论了高血压和冠心病的预防策略...")
    
    results = search.search_similar("肺癌治疗", top_k=3)
    print("搜索结果:")
    for r in results:
        print(f"  文献ID: {r['literature_id']}, 相似度: {r['similarity']:.4f}")
