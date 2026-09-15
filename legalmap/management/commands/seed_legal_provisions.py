"""
Seeds LegalProvision with real, verified Philippine legal provisions.

IMPORTANT: every entry here should be independently re-verified before your
defense — laws get amended, bills get passed/killed, and this was compiled
from a search on 22 July 2026. In particular: as of that date, the
Philippines has NO enacted law specifically addressing deepfakes. Several
bills are pending (marked PENDING_BILL below) but none had passed. If any
have since become law, move that row to ENACTED and cite the RA number.

Run with: python manage.py seed_legal_provisions
Safe to re-run — uses get_or_create keyed on (law_number, section).
"""

from django.core.management.base import BaseCommand

from legalmap.models import LegalProvision


PROVISIONS = [
    {
        "law_name": "Cybercrime Prevention Act of 2012",
        "law_number": "RA 10175",
        "section": "Sec. 4(b)(3)",
        "description": (
            "Computer-related Identity Theft: the intentional acquisition, use, misuse, "
            "transfer, possession, alteration, or deletion of identifying information "
            "belonging to another, without right. Upheld as constitutional by the Supreme "
            "Court (Disini v. Secretary of Justice, G.R. No. 203335, 2014). Penalty: "
            "prision mayor or a fine, escalating if committed against critical infrastructure."
        ),
        "status": LegalProvision.Status.ENACTED,
    },
    {
        "law_name": "Cybercrime Prevention Act of 2012",
        "law_number": "RA 10175",
        "section": "Sec. 4(c)(4)",
        "description": (
            "Cyberlibel: libel as defined under Article 355 of the Revised Penal Code, "
            "committed through a computer system. Relevant when a manipulated image is "
            "used to defame or falsely attribute statements/conduct to an identifiable person."
        ),
        "status": LegalProvision.Status.ENACTED,
    },
    {
        "law_name": "Data Privacy Act of 2012",
        "law_number": "RA 10173",
        "section": "Sec. 25",
        "description": (
            "Unauthorized Processing of Personal Information: processing personal or "
            "sensitive personal information without consent or legal basis. Relevant when "
            "a person's likeness/biometric data is used to generate synthetic media without "
            "authorization."
        ),
        "status": LegalProvision.Status.ENACTED,
    },
    {
        "law_name": "Anti-Photo and Video Voyeurism Act of 2009",
        "law_number": "RA 9995",
        "section": "Sec. 4",
        "description": (
            "Prohibits capturing, copying, selling, or distributing photos/videos of a "
            "person's private parts or sexual act without consent. Philippine courts and "
            "commentators have flagged this as one of the closest-fitting existing statutes "
            "for non-consensual sexual deepfakes, pending passage of deepfake-specific law."
        ),
        "status": LegalProvision.Status.ENACTED,
    },
    {
        "law_name": "Revised Penal Code (Estafa)",
        "law_number": "Act No. 3815",
        "section": "Art. 315",
        "description": (
            "Swindling/Estafa: defrauding another through false pretenses or fraudulent "
            "acts. Relevant when a manipulated image (e.g. fabricated ID, falsified "
            "endorsement) is used to induce financial loss or fraudulent transactions."
        ),
        "status": LegalProvision.Status.ENACTED,
    },
    {
        "law_name": "Deepfake Regulation Act",
        "law_number": "HB 3214",
        "section": "",
        "description": (
            "Proposed act prohibiting the use of deepfakes without the prior written "
            "consent of the person whose likeness is copied; would create a DICT office to "
            "receive complaints, validate identities, and issue takedown orders. Filed by "
            "Rep. Brian Raymund Yamsuan. PENDING as of July 2026 — not yet law."
        ),
        "status": LegalProvision.Status.PENDING_BILL,
    },
    {
        "law_name": "Deepfake Regulation and Digital Identity Protection Act",
        "law_number": "SB 758",
        "section": "",
        "description": (
            "Proposed act establishing that a person has exclusive control over their "
            "image, voice, and identity; treats unauthorized deepfakes as a privacy/dignity "
            "violation with a right to compensation, and mandates AI-generated-content "
            "disclosure. Filed by Sen. Bam Aquino. PENDING as of July 2026 — not yet law."
        ),
        "status": LegalProvision.Status.PENDING_BILL,
    },
    {
        "law_name": "Take It Down Act of 2025",
        "law_number": "HB 807",
        "section": "",
        "description": (
            "Proposed act penalizing creation/distribution of AI-generated sexually "
            "explicit material (up to 12 years imprisonment, 20 if victim is a minor); "
            "victims may also sue for damages. PENDING as of July 2026 — not yet law."
        ),
        "status": LegalProvision.Status.PENDING_BILL,
    },
]


class Command(BaseCommand):
    help = "Seeds LegalProvision with verified Philippine legal provisions relevant to synthetic media."

    def handle(self, *args, **options):
        created_count = 0
        updated_count = 0

        for entry in PROVISIONS:
            obj, created = LegalProvision.objects.update_or_create(
                law_number=entry["law_number"],
                section=entry["section"],
                defaults={
                    "law_name": entry["law_name"],
                    "description": entry["description"],
                    "status": entry["status"],
                },
            )
            if created:
                created_count += 1
            else:
                updated_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded legal provisions: {created_count} created, {updated_count} updated."
            )
        )