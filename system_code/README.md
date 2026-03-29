# 医学文献智能检索系统

基于自然语言处理的医疗文献智能检索系统，使用Django框架和BERT预训练模型。

## 功能特性

### 1. 数据获取与预处理
- 从公开期刊获取OA医学文献
- 文本清洗、规范化、分词处理

### 2. 文本数据预处理
- 去除无关符号
- 文本规范化
- 中文分词处理

### 3. 构建医学文献分类模型
- 基于预训练语言模型BERT
- 对标题和摘要进行语义表示
- 模型微调实现自动分类

### 4. 实现医学文献检索功能
- 关键词检索
- 语义检索（基于文本向量）
- 文本向量表示和相似度计算

### 5. 模型评估与系统测试
- 准确率、召回率、F1值评估
- 系统功能和检索速度测试

## 系统功能

1. **关键词检索** - 精确匹配标题、摘要、关键词
2. **语义检索** - 基于BERT模型理解查询意图
3. **文献主题分类** - 7大医学主题自动分类
4. **相似文献推荐** - 基于向量相似度推荐
5. **自动摘要** - 智能提取文献核心内容

## 主题分类

- 肿瘤学
- 内分泌代谢
- 心血管
- 神经内科
- 感染性疾病
- 中医药
- 其他

## 项目结构

```
system_code/
├── medical_search/          # Django项目配置
│   ├── settings.py
│   └── urls.py
├── literature/              # 文献管理应用
│   ├── models.py           # 数据模型
│   ├── views.py            # 视图函数
│   ├── serializers.py      # API序列化器
│   └── urls.py             # URL路由
├── ml_models/              # 机器学习模块
│   ├── bert_classifier.py # BERT分类器
│   ├── text_vectorizer.py  # 文本向量化
│   ├── similarity_search.py# 相似度检索
│   └── summarizer.py       # 摘要生成
├── templates/              # HTML模板
├── static/                 # 静态文件
├── manage.py
└── requirements.txt
```

## 安装与运行

### 1. 安装依赖

```bash
cd system_code
pip install -r requirements.txt
```

### 2. 数据库迁移

```bash
python manage.py makemigrations
python manage.py migrate
```

### 3. 导入数据

```bash
python import_data.py
```

### 4. 运行服务器

```bash
python manage.py runserver
```

访问 http://127.0.0.1:8000 查看系统

## API接口

### 关键词检索
```
POST /api/search/
Body: {"query": "肺癌治疗", "page": 1}
```

### 语义检索
```
POST /api/semantic-search/
Body: {"query": "关于糖尿病的最新治疗方法", "top_k": 20}
```

### 文本分类
```
POST /api/classify/
Body: {"text": "文献标题和摘要内容..."}
```

### 自动摘要
```
POST /api/summary/
Body: {"text": "长文本内容..."} 或 {"literature_id": 1}
```

### 相似文献
```
POST /api/similar/
Body: {"literature_id": 1, "top_k": 5}
```

### 批量分类
```
POST /api/auto-classify/
```

## 技术栈

- **后端**: Django 4.2, Django REST Framework
- **前端**: HTML5, CSS3, JavaScript
- **机器学习**: PyTorch, Transformers, BERT
- **数据库**: SQLite
- **中文分词**: Jieba

## 环境要求

- Python 3.8+
- CUDA (可选，用于GPU加速)
- 内存: 8GB+
- 硬盘: 10GB+

## 使用说明

### 首页
访问系统首页，可以进行快速检索和浏览最新文献。

### 文献检索
1. 选择检索类型（关键词/语义）
2. 输入检索词
3. 查看检索结果
4. 可对单篇文献进行相似文献推荐和摘要生成

### 主题分类
1. 输入文献标题和摘要
2. 点击分类获取分类结果
3. 支持批量自动分类未分类文献

### 数据导入
1. 单条导入：填写文献信息表单
2. 批量导入：粘贴JSON格式数据

### 模型评估
查看各分类的准确率、召回率、F1值，以及系统性能测试结果。

## 注意事项

1. 首次运行需要下载BERT模型（约400MB）
2. 语义检索首次使用会建立向量索引
3. 建议使用Chrome或Firefox浏览器

## 许可证

MIT License

## 作者

基于自然语言处理的医疗文献智能检索系统的设计与实现
