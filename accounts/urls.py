from django.urls import path

from .views import LogoutAPIView, RegisterAPIView

urlpatterns = [
    path("register/", RegisterAPIView.as_view(), name="account-register"),
    path("logout/", LogoutAPIView.as_view(), name="account-logout"),
]
