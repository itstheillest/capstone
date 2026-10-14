from django.urls import path

from legalmap.views import LegalMappingListView, LegalProvisionListView, LegalReportView

urlpatterns = [
    path("legalmap/report/<uuid:submission_id>/", LegalReportView.as_view(), name="legalmap-report"),
    path("legalmap/mappings/<uuid:submission_id>/", LegalMappingListView.as_view(), name="legalmap-mappings"),
    path("legalmap/provisions/", LegalProvisionListView.as_view(), name="legalmap-provisions"),
]