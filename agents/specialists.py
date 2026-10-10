class ClinicalAnalysisAgent:
    """Coordinates symptom and multimodal image analysis tools."""
    def __init__(self, tools): self.tools = tools
    def run(self, context, names):
        return [self.tools[name].run(context) for name in names if name in self.tools]

class KnowledgeResearchAgent:
    """Applies the RAG-first, web-supplement policy."""
    def __init__(self, tools): self.tools = tools
    def run(self, context):
        rag = self.tools["medical_knowledge_search"].run(context)
        results = [rag]
        if rag.get("status") != "success" or rag.get("confidence", 0) < 0.8:
            results.append(self.tools["web_search"].run(context))
        return results

class SafetyRecommendationAgent:
    """Combines history, medication, and risk tools."""
    def __init__(self, tools): self.tools = tools
    def run(self, context, include_history=False, include_drugs=False):
        results = []
        if include_history: results.append(self.tools["patient_history_query"].run(context))
        if include_drugs: results.append(self.tools["drug_interaction_checker"].run(context))
        results.append(self.tools["risk_assessor"].run(context))
        return results
