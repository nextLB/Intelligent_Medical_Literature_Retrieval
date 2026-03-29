#!/usr/bin/env python3
from rest_framework import serializers
from .models import Literature, LiteratureCategory, SearchHistory, ModelEvaluation


class LiteratureCategorySerializer(serializers.ModelSerializer):
    literature_count = serializers.SerializerMethodField()
    
    class Meta:
        model = LiteratureCategory
        fields = ['id', 'name', 'name_en', 'description', 'literature_count']
    
    def get_literature_count(self, obj):
        return obj.literatures.count()


class LiteratureListSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    
    class Meta:
        model = Literature
        fields = [
            'id', 'pmid', 'title', 'abstract', 'keywords',
            'journal', 'publish_year', 'category', 'category_name',
            'created_at'
        ]


class LiteratureDetailSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    similar_literatures = serializers.SerializerMethodField()
    
    class Meta:
        model = Literature
        fields = [
            'id', 'pmid', 'title', 'abstract', 'keywords',
            'authors', 'journal', 'publish_year', 'doi',
            'language', 'category', 'category_name',
            'similar_literatures', 'created_at', 'updated_at'
        ]
    
    def get_similar_literatures(self, obj):
        from .models import SimilarLiterature
        similar = SimilarLiterature.objects.filter(source=obj).order_by('-similarity_score')[:5]
        return [{
            'id': s.similar.id,
            'title': s.similar.title,
            'journal': s.similar.journal,
            'publish_year': s.similar.publish_year,
            'similarity_score': s.similarity_score
        } for s in similar]


class SearchHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = SearchHistory
        fields = ['id', 'query', 'search_type', 'results_count', 'created_at']


class ModelEvaluationSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    
    class Meta:
        model = ModelEvaluation
        fields = [
            'id', 'model_name', 'category', 'category_name',
            'accuracy', 'precision', 'recall', 'f1_score',
            'support', 'created_at'
        ]
