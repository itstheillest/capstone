'''
from rest_framework.routers import DefaultRouter

from .views import ImageSubmissionViewSet

router = DefaultRouter()
router.register("submissions", ImageSubmissionViewSet, basename="submission")

urlpatterns = router.urls
'''
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ImageSubmissionViewSet

router = DefaultRouter()
router.register(r'submissions', ImageSubmissionViewSet, basename='imagesubmission')

urlpatterns = [
    path('', include(router.urls)),
]