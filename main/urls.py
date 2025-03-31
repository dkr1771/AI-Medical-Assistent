from django.urls import path
from .views import *

urlpatterns = [
    path('',LoginView.as_view(),name='login'),
    path('register/',RegView.as_view(),name='reg'),
    path('home/',MainPage.as_view(),name='main'),
    path('prediction-page/',Prediction.as_view(),name='prediction'),
    path('predict/',predict,name='predict'),
    path('history/',HistoryView.as_view(),name='history'),
    path('logout/',custom_logout,name='logout'),
    path('remove/<int:pk>/',remove_History,name='del'),
    path('error_login/', error_login, name='error_login'),
]
