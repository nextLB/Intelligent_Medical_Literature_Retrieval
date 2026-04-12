#!/usr/bin/env python3
from django.db import models
from django.contrib.auth.models import User


class LiteratureCategory(models.Model):
    """文献分类"""
    name = models.CharField('分类名称', max_length=50, unique=True)
    name_en = models.CharField('英文名称', max_length=50, blank=True)
    description = models.TextField('描述', blank=True)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    
    class Meta:
        verbose_name = '文献分类'
        verbose_name_plural = '文献分类'
        ordering = ['id']
    
    def __str__(self):
        return self.name


class Literature(models.Model):
    """医学文献"""
    pmid = models.CharField('PMID', max_length=20, blank=True, unique=True)
    title = models.CharField('标题', max_length=500)
    abstract = models.TextField('摘要', blank=True)
    keywords = models.CharField('关键词', max_length=500, blank=True)
    authors = models.TextField('作者', blank=True)
    journal = models.CharField('期刊', max_length=200, blank=True)
    publish_year = models.IntegerField('发表年份', null=True, blank=True)
    doi = models.CharField('DOI', max_length=100, blank=True)
    language = models.CharField('语言', max_length=20, default='zh')
    
    category = models.ForeignKey(
        LiteratureCategory, 
        on_delete=models.SET_NULL,
        null=True, 
        blank=True,
        related_name='literatures',
        verbose_name='分类'
    )
    
    text_vector = models.TextField('文本向量', blank=True)
    text_length = models.IntegerField('文本长度', default=0)
    
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)
    
    class Meta:
        verbose_name = '医学文献'
        verbose_name_plural = '医学文献'
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title[:50]
    
    def save(self, *args, **kwargs):
        if not self.text_length:
            self.text_length = len(self.title) + len(self.abstract)
        super().save(*args, **kwargs)


class SearchHistory(models.Model):
    """搜索历史"""
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        null=True, 
        blank=True,
        related_name='search_histories',
        verbose_name='用户'
    )
    query = models.CharField('搜索词', max_length=500)
    search_type = models.CharField(
        '搜索类型', 
        max_length=20,
        choices=[
            ('keyword', '关键词搜索'),
            ('semantic', '语义搜索'),
            ('category', '分类搜索'),
        ]
    )
    results_count = models.IntegerField('结果数量', default=0)
    ip_address = models.GenericIPAddressField('IP地址', null=True, blank=True)
    created_at = models.DateTimeField('搜索时间', auto_now_add=True)
    
    class Meta:
        verbose_name = '搜索历史'
        verbose_name_plural = '搜索历史'
        ordering = ['-created_at']


class SimilarLiterature(models.Model):
    """相似文献关系"""
    source = models.ForeignKey(
        Literature,
        on_delete=models.CASCADE,
        related_name='similar_from',
        verbose_name='源文献'
    )
    similar = models.ForeignKey(
        Literature,
        on_delete=models.CASCADE,
        related_name='similar_to',
        verbose_name='相似文献'
    )
    similarity_score = models.FloatField('相似度', default=0.0)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    
    class Meta:
        verbose_name = '相似文献'
        verbose_name_plural = '相似文献'
        unique_together = ['source', 'similar']


class UserProfile(models.Model):
    """用户扩展信息"""
    ROLE_CHOICES = [
        ('user', '普通用户'),
        ('admin', '管理员'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField('角色', max_length=20, choices=ROLE_CHOICES, default='user')
    phone = models.CharField('手机', max_length=20, blank=True)
    avatar = models.ImageField('头像', upload_to='avatars/', blank=True, null=True)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)
    
    class Meta:
        verbose_name = '用户信息'
        verbose_name_plural = '用户信息'
    
    def __str__(self):
        return f"{self.user.username} ({self.get_role_display()})"


class ModelEvaluation(models.Model):
    """模型评估结果"""
    model_name = models.CharField('模型名称', max_length=100)
    category = models.ForeignKey(
        LiteratureCategory,
        on_delete=models.CASCADE,
        related_name='evaluations',
        verbose_name='分类'
    )
    accuracy = models.FloatField('准确率')
    precision = models.FloatField('精确率')
    recall = models.FloatField('召回率')
    f1_score = models.FloatField('F1值')
    support = models.IntegerField('样本数')
    created_at = models.DateTimeField('评估时间', auto_now_add=True)
    
    class Meta:
        verbose_name = '模型评估'
        verbose_name_plural = '模型评估'
        ordering = ['-created_at']
