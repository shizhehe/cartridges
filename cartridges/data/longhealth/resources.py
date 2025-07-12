from typing import Dict, List, Optional, Tuple, Any

from cartridges.utils import get_logger
from cartridges.data.resources import BaseStructuredResource
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