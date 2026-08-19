import logging
from celery import shared_task
from django.db import transaction
from django.utils import timezone

from .models import ImageSubmission, ReverseImageMatch, FactCheckReference
from .services.vision_service import vision_service
from .services.factcheck_service import factcheck_service

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=10)
def process_image_submission(self, submission_id: int):
    """
    Celery task to handle asynchronous image analysis:
    1. Update status to PROCESSING.
    2. Call Google Vision API & store ReverseImageMatch entries.
    3. Call Fact Check API & store FactCheckReference entries.
    4. Update status to COMPLETED (or FAILED if error occurs).
    """
    try:
        submission = ImageSubmission.objects.get(pk=submission_id)
    except ImageSubmission.DoesNotExist:
        logger.error(f"ImageSubmission ID {submission_id} does not exist.")
        return

    # Update status to PROCESSING
    submission.status = ImageSubmission.Status.PROCESSING
    submission.save(update_fields=['status'])

    try:
        # Step 1: Run Google Vision API Search
        vision_results = vision_service.search_image(submission.image.path)
        
        # Step 2: Run Fact Check API Search using extracted queries or labels
        queries = vision_results.get("web_entities", [])
        factcheck_results = []
        if queries:
            # Query Fact Check API using the top detected entity/label
            top_query = queries[0].get("description", "") if isinstance(queries[0], dict) else str(queries[0])
            if top_query:
                factcheck_results = factcheck_service.search_claims(top_query)

        # Step 3: Atomic Database Orchestration
        with transaction.atomic():
            # Clear old results if retrying
            ReverseImageMatch.objects.filter(submission=submission).delete()
            FactCheckReference.objects.filter(submission=submission).delete()

            # Store Reverse Image Matches
            matches_to_create = []
            for match in vision_results.get("full_matching_images", []):
                matches_to_create.append(
                    ReverseImageMatch(
                        submission=submission,
                        url=match.get("url", ""),
                        title=match.get("title", ""),
                        source_domain=match.get("domain", ""),
                        match_type=ReverseImageMatch.MatchType.EXACT,
                        score=match.get("score")
                    )
                )
            for match in vision_results.get("partial_matching_images", []):
                matches_to_create.append(
                    ReverseImageMatch(
                        submission=submission,
                        url=match.get("url", ""),
                        title=match.get("title", ""),
                        source_domain=match.get("domain", ""),
                        match_type=ReverseImageMatch.MatchType.PARTIAL,
                        score=match.get("score")
                    )
                )
            if matches_to_create:
                ReverseImageMatch.objects.bulk_create(matches_to_create)

            # Store Fact Check References
            references_to_create = []
            for claim in factcheck_results:
                references_to_create.append(
                    FactCheckReference(
                        submission=submission,
                        claim_text=claim.get("text", ""),
                        claimant=claim.get("claimant", ""),
                        rating=claim.get("rating", ""),
                        url=claim.get("url", ""),
                        publisher=claim.get("publisher", "")
                    )
                )
            if references_to_create:
                FactCheckReference.objects.bulk_create(references_to_create)

            # Mark status as COMPLETED
            submission.status = ImageSubmission.Status.COMPLETED
            submission.processed_at = timezone.now()
            submission.save(update_fields=['status', 'processed_at'])

    except Exception as exc:
        logger.exception(f"Error processing submission {submission_id}: {exc}")
        
        # Mark status as FAILED
        submission.status = ImageSubmission.Status.FAILED
        submission.save(update_fields=['status'])
        
        # Retry celery task if under max retries
        raise self.retry(exc=exc)