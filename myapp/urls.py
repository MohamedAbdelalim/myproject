from django.urls import path
from .views import PlateRecognitionAPIView

urlpatterns = [
    path('recognize/', PlateRecognitionAPIView.as_view(), name='recognize_plate'),
]