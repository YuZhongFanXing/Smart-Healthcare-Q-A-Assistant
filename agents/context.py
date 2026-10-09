from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class TaskContext:
    message: str = ""
    image: Any = None
    modality: Optional[str] = None
    common_metadata: Dict[str, Any] = field(default_factory=dict)
    modality_metadata: Dict[str, Any] = field(default_factory=dict)
    patient_id: Optional[str] = None
    session_id: Optional[str] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    def missing_fields(self):
        required = {"skin": ["age", "sex", "anatom_site"], "oral": ["age", "sex", "oral_site"]}
        values = {**self.common_metadata, **self.modality_metadata}
        return [field for field in required.get(self.modality, [])
                if not values.get(field) or (field == "sex" and values.get(field) == "unknown")]
