from rest_framework.routers import DefaultRouter

from .views import ImageSubmissionViewSet

router = DefaultRouter()
router.register("submissions", ImageSubmissionViewSet, basename="submission")

urlpatterns = router.urls