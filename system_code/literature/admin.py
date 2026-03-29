#!/usr/bin/env python3
from django.contrib import admin
from .models import Literature, LiteratureCategory, SearchHistory, SimilarLiterature, ModelEvaluation


@admin.register(LiteratureCategory)
class LiteratureCategoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'name_en', 'created_at']
    search_fields = ['name', 'name_en']


@admin.register(Literature)
class LiteratureAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'category', 'journal', 'publish_year', 'created_at']
    list_filter = ['category', 'publish_year', 'language']
    search_fields = ['title', 'abstract', 'keywords', 'pmid']
    raw_id_fields = ['category']


@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'query', 'search_type', 'results_count', 'created_at']
    list_filter = ['search_type', 'created_at']
    search_fields = ['query']


@admin.register(SimilarLiterature)
class SimilarLiteratureAdmin(admin.ModelAdmin):
    list_display = ['id', 'source', 'similar', 'similarity_score', 'created_at']
    list_filter = ['created_at']


@admin.register(ModelEvaluation)
class ModelEvaluationAdmin(admin.ModelAdmin):
    list_display = ['id', 'model_name', 'category', 'accuracy', 'precision', 'recall', 'f1_score', 'created_at']
    list_filter = ['model_name', 'category', 'created_at']
