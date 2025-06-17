from typing import List, Optional
from pydantic import BaseModel

from cartridges.context import TexDocument, TexChapter, TexSection
from cartridges.context import BaseContextConfig

from cartridges.context import StructuredContext
from cartridges.tasks.longhealth.load import load_longhealth_dataset, LongHealthPatient, LongHealthQuestion

class LongHealthStructuredContextConfig(BaseContextConfig):
    patient_ids: Optional[List[str]] = None
    
    def instantiate(self) -> StructuredContext:
        patients = load_longhealth_dataset(self.patient_ids)

        return PatientPanel(
            patients=[
                Patient(
                    patient_id=patient.patient_id,
                    notes=[
                        MedicalNote(note_id=note_id, content=note) 
                        for note_id, note in patient.texts.items()
                    ],
                    name=patient.name,
                    birthday=patient.birthday,
                    diagnosis=patient.diagnosis,
                )
                for patient in patients
            ]
        )


class MedicalNote(StructuredContext):
    note_id: str
    content: str


class Patient(StructuredContext):
    patient_id: str
    notes: List[MedicalNote]
    name: str 
    birthday: str
    diagnosis: str


class PatientPanel(StructuredContext):
    patients: List[Patient]