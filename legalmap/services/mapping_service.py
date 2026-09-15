"""
Law Mapping Layer.

Looks up which LegalProvision rows a given tag_code (from the Tagging
Layer) implicates, and creates LegalMapping rows linking them to a
DetectionReport. This is deliberately a static lookup table, not something
inferred at runtime by a model — so a reviewer (or your panel) can audit
exactly which rule produced which legal citation, and so a lawyer can
review/edit TAG_TO_PROVISIONS without touching detection code.

Update TAG_TO_PROVISIONS as your Tagging Layer's rule set grows in
detection/services/tagging_service.py — tag_code values must match exactly.
"""

from legalmap.models import LegalMapping, LegalProvision

TAG_TO_PROVISIONS = {
    "identity_fraud": [
        ("RA 10175", "Sec. 4(b)(3)"),   # Computer-related Identity Theft
        ("Act No. 3815", "Art. 315"),   # Estafa
        ("HB 3214", ""),                # Deepfake Regulation Act (pending)
        ("SB 758", ""),                 # Deepfake Regulation and Digital Identity Protection Act (pending)
    ],
    "manipulated_media": [
        ("RA 10175", "Sec. 4(c)(4)"),   # Cyberlibel
        ("HB 3214", ""),                # Deepfake Regulation Act (pending)
    ],
}


def apply_mappings(report, tag_codes: list[str]) -> list[LegalMapping]:
    """
    Creates (or refreshes) LegalMapping rows for a DetectionReport based on
    its submission's tag codes. Returns the list of LegalMapping rows created.
    """
    LegalMapping.objects.filter(report=report, trigger_condition__in=TAG_TO_PROVISIONS.keys()).delete()

    created = []
    for tag_code in tag_codes:
        provision_keys = TAG_TO_PROVISIONS.get(tag_code)
        if not provision_keys:
            continue

        for law_number, section in provision_keys:
            try:
                provision = LegalProvision.objects.get(law_number=law_number, section=section)
            except LegalProvision.DoesNotExist:
                continue

            mapping, _ = LegalMapping.objects.get_or_create(
                report=report,
                provision=provision,
                trigger_condition=tag_code,
                defaults={
                    "notes": f"Auto-mapped from tag '{tag_code}' by the Tagging Layer.",
                },
            )
            created.append(mapping)

    return created