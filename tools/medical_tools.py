import logging
from typing import Any

logger = logging.getLogger(__name__)

class BaseTool:
    name = "tool"
    def result(self, status="success", **data):
        return {"tool": self.name, "status": status, **data}

class SkinLesionClassifier(BaseTool):
    name = "skin_lesion_classifier"
    def __init__(self, predictor): self.predictor = predictor
    def run(self, context):
        if not self.predictor: return self.result("unavailable", error="skin model is not loaded")
        m = {**context.common_metadata, **context.modality_metadata}
        prediction = self.predictor.predict(context.image, m.get("age"), m.get("sex"), m.get("anatom_site"))
        status = prediction.pop("status", "success")
        return self.result(status, **prediction)

class OralLesionDetector(BaseTool):
    name = "oral_lesion_detector"
    def run(self, context):
        return self.result("unavailable", error="oral lesion model provider is not configured", requires=["oral_model_provider"])

class SymptomAnalyzer(BaseTool):
    name = "symptom_analyzer"
    def run(self, context):
        text = context.message.lower()
        hints = [k for k in ("疼", "痛", "痒", "出血", "发热", "溃疡") if k in text]
        return self.result(symptoms=hints, note="Rule-based placeholder; replace with model provider.")

class MedicalKnowledgeSearch(BaseTool):
    name = "medical_knowledge_search"
    def run(self, context):
        from config import RAGFLOW_AUTHORIZATION, RAGFLOW_CHAT_ID
        if not RAGFLOW_AUTHORIZATION or not RAGFLOW_CHAT_ID:
            return self.result(
                answer="[模拟医学知识库] 已根据当前问题检索内部医学知识。RAGFlow 尚未配置，当前返回形式化结果。",
                confidence=0.65,
                source_type="rag",
                sources=[{"title": "模拟医学知识库", "source": "local-placeholder"}],
                mocked=True,
            )
        try:
            from services.ragflow_service import stream_chat
            answer, session = stream_chat(context.message, context.session_id)
            return self.result(answer=answer, confidence=0.7, source_type="rag", session_id=session)
        except Exception as exc:
            return self.result("unavailable", error=str(exc), source_type="rag")

class WebSearch(BaseTool):
    name = "web_search"
    def run(self, context):
        return self.result(
            "success",
            answer="[模拟 Web Search] 当前未配置外部搜索 API，已返回形式化检索结果。",
            source_type="web",
            sources=[{"title": "模拟权威来源", "source": "web-placeholder", "date": None}],
            mocked=True,
        )

class RiskAssessor(BaseTool):
    name = "risk_assessor"
    def run(self, context):
        results = context.tool_results
        confidence = max((float(r.get("confidence", 0)) for r in results if r.get("status") == "success"), default=0)
        level = "high" if confidence and confidence < 0.5 else "medium" if confidence else "unknown"
        return self.result(risk_level=level, confidence=confidence)

class DrugInteractionChecker(BaseTool):
    name = "drug_interaction_checker"
    def run(self, context):
        return self.result("unavailable", error="drug database provider is not configured")

class PatientHistoryQuery(BaseTool):
    name = "patient_history_query"
    def run(self, context):
        return self.result("unavailable", error="patient history provider is not configured")
