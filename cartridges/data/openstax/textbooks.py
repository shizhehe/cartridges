
from capsules.data.openstax.structs import TextbookMetadata


BUSINESS_LAW = TextbookMetadata(
    github_repo="osbooks-business-law",
    collection_name="business-law-i-essentials.collection.xml",
    openstax_url="https://openstax.org/books/business-law-i-essentials/pages",
    review_questions_types=["assessment-questions"],
)

NURSE_PHARM = TextbookMetadata(
    github_repo="osbooks-nursing-external-bundle",
    collection_name="pharmacology.collection.xml",
    openstax_url="https://openstax.org/books/pharmacology/pages",
    review_questions_types=["review-questions"],
)