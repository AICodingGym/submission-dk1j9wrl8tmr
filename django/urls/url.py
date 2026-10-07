# filepath: /Users/isabellajiang/django-15957/django/urls.py
from django.urls import path
from .views.views import category_list_view

urlpatterns = [
    path('categories/', category_list_view, name='category_list'),
]