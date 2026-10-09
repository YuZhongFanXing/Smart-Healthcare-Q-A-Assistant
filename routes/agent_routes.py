from flask import Blueprint, request, jsonify
from agents.context import TaskContext

agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")
_agent = None
_contexts = {}

def init_agent(agent):
    global _agent
    _agent = agent

@agent_bp.route("/chat", methods=["POST"])
def agent_chat():
    if _agent is None: return jsonify({"status": "error", "error": "agent is not initialized"}), 503
    data = request.form
    image = request.files.get("image") or (request.files.getlist("images[]") or [None])[0]
    metadata = {}
    if data.get("metadata"):
        import json
        try: metadata = json.loads(data.get("metadata"))
        except ValueError: return jsonify({"status": "error", "error": "metadata must be JSON"}), 400
    for key in ("age", "sex", "anatom_site", "oral_site"):
        if data.get(key): metadata[key] = data.get(key)
    session_id = data.get("session_id") or "anonymous"
    previous = _contexts.get(session_id)
    if previous and not image:
        image = previous.image
    context = TaskContext(message=data.get("message", ""), image=image, modality=data.get("modality") or (previous.modality if previous else None),
                          common_metadata={k: metadata[k] for k in ("age", "sex") if k in metadata},
                          modality_metadata={k: v for k, v in metadata.items() if k not in ("age", "sex")},
                          patient_id=data.get("patient_id") or (previous.patient_id if previous else None), session_id=session_id)
    if previous:
        context.common_metadata = {**previous.common_metadata, **context.common_metadata}
        context.modality_metadata = {**previous.modality_metadata, **context.modality_metadata}
    result = _agent.run(context)
    _contexts[session_id] = context
    result["session_id"] = session_id
    return jsonify(result)
