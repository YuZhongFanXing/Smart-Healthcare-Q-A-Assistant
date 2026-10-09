import logging
from .medical_tools import (
    SkinLesionClassifier, OralLesionDetector, SymptomAnalyzer,
    MedicalKnowledgeSearch, WebSearch, RiskAssessor,
    DrugInteractionChecker, PatientHistoryQuery,
)

logger = logging.getLogger(__name__)

def build_registry(predictor=None):
    return {
        "skin_lesion_classifier": SkinLesionClassifier(predictor),
        "oral_lesion_detector": OralLesionDetector(),
        "symptom_analyzer": SymptomAnalyzer(),
        "medical_knowledge_search": MedicalKnowledgeSearch(),
        "web_search": WebSearch(),
        "risk_assessor": RiskAssessor(),
        "drug_interaction_checker": DrugInteractionChecker(),
        "patient_history_query": PatientHistoryQuery(),
    }
