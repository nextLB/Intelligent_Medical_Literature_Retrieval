#!/usr/bin/env python3
import jieba
import re
from collections import Counter


class TextSummarizer:
    def __init__(self):
        jieba.setLogLevel(jieba.logging.INFO)
        self.stopwords = self._load_stopwords()
        self.english_stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too',
            'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once'
        ])
    
    def _load_stopwords(self):
        stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '为', '之', '以', '而', '于', '及', '与', '或', '等',
            '该', '其', '中', '对', '可', '能', '下', '过', '来', '他', '她', '它', '他们',
            '我们', '什么', '怎么', '这样', '那样', '因为', '所以', '但是', '虽然', '如果'
        ])
        return stopwords
    
    def _is_english(self, text):
        english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return english_chars > chinese_chars * 2
    
    def extractive_summarize(self, text, top_n=3):
        is_english = self._is_english(text)
        
        if is_english:
            return self._english_summarize(text, top_n)
        else:
            return self._chinese_summarize(text, top_n)
    
    def _english_summarize(self, text, top_n=3):
        sentences = re.split(r'[.!?\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= top_n:
            return text
        
        word_scores = self._calculate_english_word_scores(sentences)
        sentence_scores = self._calculate_sentence_scores(sentences, word_scores)
        
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[0]))
        top_sentences = [s[0] for s in top_sentences]
        
        return '. '.join(top_sentences) + '.'
    
    def _chinese_summarize(self, text, top_n=3):
        sentences = self._split_sentences(text)
        if len(sentences) <= top_n:
            return text
        
        word_scores = self._calculate_word_scores(sentences)
        sentence_scores = self._calculate_sentence_scores(sentences, word_scores)
        
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[0]))
        top_sentences = [s[0] for s in top_sentences]
        
        return '。'.join(top_sentences) + '。'
    
    def _split_sentences(self, text):
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _calculate_word_scores(self, sentences):
        word_freq = Counter()
        for sentence in sentences:
            words = jieba.cut(sentence)
            words = [w for w in words if w not in self.stopwords and len(w) > 1]
            word_freq.update(words)
        
        max_freq = max(word_freq.values()) if word_freq else 1
        
        word_scores = {}
        for word, freq in word_freq.items():
            word_scores[word] = freq / max_freq
        
        return word_scores
    
    def _calculate_english_word_scores(self, sentences):
        word_freq = Counter()
        for sentence in sentences:
            words = re.findall(r'[a-zA-Z]+', sentence.lower())
            words = [w for w in words if w not in self.english_stopwords and len(w) > 2]
            word_freq.update(words)
        
        max_freq = max(word_freq.values()) if word_freq else 1
        
        word_scores = {}
        for word, freq in word_freq.items():
            word_scores[word] = freq / max_freq
        
        return word_scores
    
    def _calculate_sentence_scores(self, sentences, word_scores):
        sentence_scores = {}
        for sentence in sentences:
            if self._is_english(sentence):
                words = re.findall(r'[a-zA-Z]+', sentence.lower())
                words = [w for w in words if w not in self.english_stopwords and len(w) > 2]
            else:
                words = jieba.cut(sentence)
                words = [w for w in words if w not in self.stopwords and len(w) > 1]
            
            if not words:
                sentence_scores[sentence] = 0
                continue
            
            score = sum(word_scores.get(w, 0) for w in words) / len(words)
            sentence_scores[sentence] = score
        
        return sentence_scores
    
    def generate_abstract(self, text, max_length=200):
        summary = self.extractive_summarize(text, top_n=3)
        
        if len(summary) > max_length:
            summary = summary[:max_length]
            if self._is_english(text):
                last_punct = summary.rfind('.')
                if last_punct > max_length * 0.5:
                    summary = summary[:last_punct + 1]
            else:
                last_punct = summary.rfind('。')
                if last_punct > max_length * 0.5:
                    summary = summary[:last_punct + 1]
        
        return summary
    
    def keyword_extraction(self, text, top_k=10):
        is_english = self._is_english(text)
        
        if is_english:
            words = re.findall(r'[a-zA-Z]+', text.lower())
            words = [w for w in words if w not in self.english_stopwords and len(w) > 2]
        else:
            words = jieba.cut(text)
            words = [w for w in words if w not in self.stopwords and len(w) > 1]
        
        word_freq = Counter(words)
        
        keywords = word_freq.most_common(top_k)
        return [kw[0] for kw in keywords]


text_summarizer = None


def get_text_summarizer():
    global text_summarizer
    if text_summarizer is None:
        text_summarizer = TextSummarizer()
    return text_summarizer


if __name__ == '__main__':
    summarizer = get_text_summarizer()
    
    sample_text = """
    目的 探讨PD-1抑制剂联合化疗在晚期非小细胞肺癌患者中的疗效和安全性。
    方法 纳入120例晚期非小细胞肺癌患者，随机分为联合组和对照组。联合组给予PD-1抑制剂联合化疗，对照组仅给予化疗。
    结果 联合组客观缓解率为55%，显著高于对照组的32%。联合组中位无进展生存期为11.2个月，对照组为6.8个月。
    结论 PD-1抑制剂联合化疗在晚期非小细胞肺癌患者中显示出良好的疗效和可接受的安全性。
    """
    
    summary = summarizer.extractive_summarize(sample_text, top_n=3)
    print("摘要:")
    print(summary)
    
    keywords = summarizer.keyword_extraction(sample_text, top_k=5)
    print("\n关键词:", keywords)
