import requests
import xml.etree.ElementTree as ET
import time
import json
from typing import List, Dict, Optional

class PubMed医学文献爬虫:
    """PubMed医学文献爬虫，使用NCBI官方E-utilities API"""
    
    基础URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: str = None, 邮箱: str = "researcher@example.com"):
        self.api_key = api_key
        self.邮箱 = 邮箱
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MedicalLiteratureResearch/1.0 (mailto:' + 邮箱 + ')'
        })
    
    def _请求(self, url: str, params: dict = None, 最大重试次数: int = 3) -> Optional[requests.Response]:
        """发送网络请求"""
        for i in range(最大重试次数):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp
                print(f"请求失败: {resp.status_code}, 重试 {i+1}/{最大重试次数}")
            except Exception as e:
                print(f"请求异常: {e}, 重试 {i+1}/{最大重试次数}")
            time.sleep(1)
        return None
    
    def 搜索文献(self, 查询语句: str, 最大结果数: int = 100, 起始位置: int = 0) -> List[str]:
        """搜索PubMed，返回PMID列表"""
        params = {
            'db': 'pubmed',
            'term': 查询语句,
            'retmax': 最大结果数,
            'retstart': 起始位置,
            'usehistory': 'y',
            'rettype': 'abstract'
        }
        
        url = f"{self.基础URL}/esearch.fcgi"
        resp = self._请求(url, params)
        
        if not resp:
            return []
        
        try:
            root = ET.fromstring(resp.text)
            id_list = root.find('IdList')
            if id_list is not None:
                return [id_elem.text for id_elem in id_list.findall('Id')]
        except Exception as e:
            print(f"解析XML失败: {e}")
        
        return []
    
    def 获取详情(self, pmids: List[str]) -> List[Dict]:
        """获取文献详细信息"""
        if not pmids:
            return []
        
        id_str = ','.join(pmids)
        params = {
            'db': 'pubmed',
            'id': id_str,
            'rettype': 'xml',
            'retmode': 'xml'
        }
        
        url = f"{self.基础URL}/esummary.fcgi"
        resp = self._请求(url, params)
        
        if not resp:
            return []
        
        论文列表 = []
        try:
            root = ET.fromstring(resp.text)
            for doc_sum in root.findall('.//DocSum'):
                论文 = {'pmid': '', '标题': '', '作者': [], 
                        '期刊': '', '发布日期': '', '摘要': ''}
                
                pmid_elem = doc_sum.find("Id")
                if pmid_elem is not None:
                    论文['pmid'] = pmid_elem.text
                
                for item in doc_sum.findall('Item'):
                    name = item.get('Name')
                    if name == 'Title':
                        论文['标题'] = item.text or ''
                    elif name == 'FullJournalName':
                        论文['期刊'] = item.text or ''
                    elif name == 'PubDate':
                        论文['发布日期'] = item.text or ''
                    elif name == 'AuthorList':
                        论文['作者'] = [a.text for a in item.findall('Author') if a.text]
                
                论文列表.append(论文)
        except Exception as e:
            print(f"解析详情失败: {e}")
        
        return 论文列表
    
    def 获取摘要(self, pmids: List[str]) -> Dict[str, str]:
        """获取文献摘要"""
        if not pmids:
            return {}
        
        id_str = ','.join(pmids)
        params = {
            'db': 'pubmed',
            'id': id_str,
            'rettype': 'abstract',
            'retmode': 'xml'
        }
        
        url = f"{self.基础URL}/efetch.fcgi"
        resp = self._请求(url, params)
        
        摘要字典 = {}
        if not resp:
            return 摘要字典
        
        try:
            root = ET.fromstring(resp.text)
            for article in root.findall('.//PubmedArticle'):
                pmid_elem = article.find('.//PMID')
                abstract_elem = article.find('.//AbstractText')
                
                if pmid_elem is not None and abstract_elem is not None:
                    摘要字典[pmid_elem.text] = abstract_elem.text or ''
        except Exception as e:
            print(f"解析摘要失败: {e}")
        
        return 摘要字典
    
    def 爬取(self, 关键词列表: List[str], 每关键词最大数量: int = 50) -> List[Dict]:
        """主爬取流程"""
        所有论文 = []
        已见PMID = set()
        
        for 关键词 in 关键词列表:
            print(f"\n===== 搜索关键词: {关键词} =====")
            pmids = self.搜索文献(关键词, 最大结果数=每关键词最大数量)
            print(f"  找到 {len(pmids)} 篇文献")
            
            for pmid in pmids:
                if pmid in 已见PMID:
                    continue
                已见PMID.add(pmid)
                time.sleep(0.4)
            
            if pmids:
                论文列表 = self.获取详情(pmids)
                摘要字典 = self.获取摘要(pmids)
                
                for 论文 in 论文列表:
                    论文['摘要'] = 摘要字典.get(论文['pmid'], '')
                    论文['搜索关键词'] = 关键词
                    所有论文.append(论文)
                
                time.sleep(1)
        
        print(f"\n爬取完成，共获取 {len(所有论文)} 篇文献")
        return 所有论文
    
    def 保存到JSON(self, 数据: List[Dict], 文件名: str = 'medical_literature.json'):
        """保存数据到JSON文件"""
        with open(文件名, 'w', encoding='utf-8') as f:
            json.dump(数据, f, ensure_ascii=False, indent=2)
        print(f"数据已保存至 {文件名}")


def main():
    关键词列表 = [
        'medical treatment',
        'clinical medicine', 
        'healthcare',
        'disease diagnosis',
        'therapeutic',
        'hospital management',
        'traditional Chinese medicine'
    ]
    
    爬虫 = PubMed医学文献爬虫()
    文献数据 = 爬虫.爬取(关键词列表, 每关键词最大数量=30)
    爬虫.保存到JSON(文献数据, 'medical_literature.json')
    
    if 文献数据:
        print("\n示例数据:")
        样本 = 文献数据[0]
        print(f"标题: {样本['标题'][:80]}...")
        print(f"PMID: {样本['pmid']}")
        print(f"期刊: {样本['期刊']}")


if __name__ == '__main__':
    main()
