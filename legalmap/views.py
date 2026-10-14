"""
LegalMap API Views — Phase 4.

Endpoints:
  GET  /api/legalmap/report/<submission_id>/
       Returns the full legal mapping summary for a completed submission.

  GET  /api/legalmap/provisions/
       Lists all seeded LegalProvision rows (useful for the frontend legend).
"""

import logging

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from detection.models import DetectionReport, ImageSubmission
from legalmap.models import LegalMapping, LegalProvision
from legalmap.serializers import LegalMappingSerializer, LegalProvisionSerializer
from legalmap.services.mapping_service import generate_report_summary

logger = logging.getLogger(__name__)


class LegalReportView(APIView):
    """
    GET /api/legalmap/report/<submission_id>/

    Returns the aggregated legal mapping summary for a submission's
    DetectionReport. The response separates enacted law from pending bills
    so the frontend can display the correct legal weight of each provision.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, submission_id):
        submission = get_object_or_404(ImageSubmission, pk=submission_id)

        # Only the submitting user or staff may view the report.
        if submission.submitted_by != request.user and not request.user.is_staff:
            return Response(
                {"detail": "You do not have permission to view this report."},
                status=status.HTTP_403_FORBIDDEN,
            )

        report = get_object_or_404(DetectionReport, submission=submission)
        summary = generate_report_summary(report)
        return Response(summary, status=status.HTTP_200_OK)


class LegalMappingListView(APIView):
    """
    GET /api/legalmap/mappings/<submission_id>/

    Returns raw LegalMapping rows (with nested provision detail) for a
    submission. Useful for the admin panel and detailed legal audit exports.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, submission_id):
        submission = get_object_or_404(ImageSubmission, pk=submission_id)

        if submission.submitted_by != request.user and not request.user.is_staff:
            return Response(
                {"detail": "You do not have permission to view this data."},
                status=status.HTTP_403_FORBIDDEN,
            )

        report = get_object_or_404(DetectionReport, submission=submission)
        mappings = LegalMapping.objects.filter(report=report).select_related("provision")
        serializer = LegalMappingSerializer(mappings, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class LegalProvisionListView(APIView):
    """
    GET /api/legalmap/provisions/

    Lists all seeded LegalProvision rows. Public endpoint so the frontend
    can display the full legal legend without requiring a submission.
    """

    def get(self, request):
        provisions = LegalProvision.objects.all()
        serializer = LegalProvisionSerializer(provisions, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
