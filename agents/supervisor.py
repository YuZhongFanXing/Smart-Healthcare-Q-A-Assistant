from .context import TaskContext
from .specialists import ClinicalAnalysisAgent, KnowledgeResearchAgent, SafetyRecommendationAgent

class SupervisorAgent:
    def __init__(self, tools):
        self.tools = tools
        self.clinical = ClinicalAnalysisAgent(tools)
        self.knowledge = KnowledgeResearchAgent(tools)
        self.safety = SafetyRecommendationAgent(tools)

    def _detect_modality(self, context):
        text = context.message.lower()
        if context.modality in ("skin", "oral"): return context.modality
        if any(x in text for x in ("口腔", "牙龈", "舌", "口内")): return "oral"
        if context.image: return None
        return None

    def run(self, context):
        context.modality = self._detect_modality(context)
        if context.image and not context.modality:
            return {"status": "success", "need_user_input": True, "questions": ["这张图片来自皮肤还是口腔？"], "tool_trace": []}
        if context.image and context.missing_fields():
            labels = {"age": "年龄", "sex": "性别", "anatom_site": "皮肤病灶部位", "oral_site": "口腔病变部位"}
            return {"status": "success", "need_user_input": True, "questions": [f"为了结合图片分析，请补充：{'、'.join(labels[x] for x in context.missing_fields())}。"], "tool_trace": []}

        selected = []
        if context.image:
            selected.append("skin_lesion_classifier" if context.modality == "skin" else "oral_lesion_detector")
        if context.message:
            selected.append("symptom_analyzer")
        context.tool_results.extend(self.clinical.run(context, selected))
        if context.message:
            context.tool_results.extend(self.knowledge.run(context))
        context.tool_results.extend(self.safety.run(context, bool(context.patient_id), any(x in context.message for x in ("药", "用药", "相互作用"))))
        risk = next((r for r in reversed(context.tool_results) if r.get("tool") == "risk_assessor"), {"risk_level": "unknown"})
        unavailable = [r["tool"] for r in context.tool_results if r.get("status") == "unavailable"]
        answer = self._compose(context, unavailable)
        return {"status": "success", "need_user_input": False, "response": answer, "tool_trace": context.tool_results,
                "risk_level": risk.get("risk_level", "unknown"), "session_id": context.session_id}

    def _compose(self, context, unavailable):
        lines = []
        for result in context.tool_results:
            if result.get("tool") == "skin_lesion_classifier" and result.get("status") == "success":
                lines.append(f"图片分析倾向于：{result.get('disease_name', result.get('prediction'))}，置信度 {result.get('confidence', 0):.1%}。")
            if result.get("tool") == "symptom_analyzer" and result.get("symptoms"):
                lines.append("已识别到症状线索：" + "、".join(result["symptoms"]) + "。")
            if result.get("tool") in ("medical_knowledge_search", "web_search") and result.get("answer"):
                lines.append(result["answer"])
        if unavailable: lines.append("以下能力尚未配置真实 provider：" + "、".join(unavailable) + "。")
        lines.append("以上内容仅供健康信息参考，不能替代医生面诊或正式诊断。")
        return "\n\n".join(lines)
