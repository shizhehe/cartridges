from typing import Dict, List, Optional, Tuple, Any
import random

from cartridges.utils import get_logger
from cartridges.data.resources import BaseStructuredResource, sample_seed_prompts
from cartridges.data.longhealth.utils import load_longhealth_dataset
logger = get_logger(__name__)


class LongHealthResource(BaseStructuredResource):
    class Config(BaseStructuredResource.Config):
        patient_ids: Optional[List[str]] = None
    
    def _load_data(self) -> List[Dict[str, Any]]:
        patients = load_longhealth_dataset(self.config.patient_ids)

        return [
            dict(
                patient_id=patient.patient_id,
                notes=[
                    dict(note_id=note_id, content=note) 
                    for note_id, note in patient.texts.items()
                ],
                name=patient.name,
                birthday=patient.birthday,
                diagnosis=patient.diagnosis,
            )
            for patient in patients
        ]
    
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        path, obj_str = random.choice(self.ctxs)
        ctx = f"Below is a section of a patient's medical record. It is part of a larger corpus of medical records for {len(self.config.patient_ids)} different patients. The identifier for the section of the record is {path}."
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts