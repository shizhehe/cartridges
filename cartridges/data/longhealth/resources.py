from typing import Dict, List, Optional, Tuple, Any
import random

from cartridges.utils import get_logger
from cartridges.data.resources import Resource, sample_seed_prompts, SEED_TYPES
from cartridges.data.longhealth.utils import load_longhealth_dataset
logger = get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
Below is a section of {name}'s medical record (ID: {patient_id}). 
They were born on {birthday} and have the following diagnosis: {diagnosis}.
The patients medical record consists of {num_notes} notes.
<{note_id}>
{text}
</{note_id}>"""

class LongHealthResource(Resource):
    class Config(Resource.Config):
        patient_ids: Optional[List[str]] = None
        seed_prompts: List[SEED_TYPES] = ["generic"]
        
    
    def __init__(self, config: Config):
        self.config = config
        self.patients = load_longhealth_dataset(self.config.patient_ids)

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        patient = random.choice(self.patients)
        note_id, text = random.choice(list(patient.texts.items()))

        
        ctx = SYSTEM_PROMPT_TEMPLATE.format(
            name=patient.name,
            patient_id=patient.patient_id,
            birthday=patient.birthday,
            diagnosis=patient.diagnosis,
            num_notes=len(patient.texts),
            note_id=note_id,
            text=text,
        )
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts