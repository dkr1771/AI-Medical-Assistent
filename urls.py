from django.urls import path
from .views import login_view, RegView, MainPage, Prediction, HistoryView, custom_logout, predict, remove_History

urlpatterns = [
    path('login/', login_view, name='login'),
    # ... other URL patterns ...
] 