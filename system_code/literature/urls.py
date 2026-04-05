#!/usr/bin/env python3
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('literatures/', views.literature_list, name='literature_list'),
    path('literature/<int:pk>/', views.literature_detail, name='literature_detail'),
    path('search/', views.search_page, name='search_page'),
    path('api/search/', views.keyword_search, name='keyword_search'),
    path('api/semantic-search/', views.semantic_search_view, name='semantic_search'),
    path('api/classify/', views.classify_literature, name='classify_literature'),
    path('api/auto-classify/', views.auto_classify_all, name='auto_classify'),
    path('api/summary/', views.generate_summary, name='generate_summary'),
    path('api/similar/', views.find_similar, name='find_similar'),
    path('api/categories/statistics/', views.CategoryStatisticsView.as_view(), name='category_statistics'),
    path('api/search-history/', views.SearchHistoryView.as_view(), name='search_history'),
    path('classification/', views.classification_page, name='classification_page'),
    path('evaluation/', views.evaluation_page, name='evaluation_page'),
    path('data-import/', views.data_import_page, name='data_import_page'),
    path('api/import/', views.import_literature, name='import_literature'),
    path('training/', views.training_page, name='training_page'),
    path('api/train/', views.train_model, name='train_model'),
    path('api/train-stream/', views.train_model_stream, name='train_stream'),
    path('api/train-stop/', views.stop_training, name='train_stop'),
    path('api/training-history/', views.get_training_history, name='training_history'),
]
