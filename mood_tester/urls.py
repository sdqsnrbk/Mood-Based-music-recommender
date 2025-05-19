# mood_tester/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('test/', views.test_mood_view, name='test_mood_page'),
]