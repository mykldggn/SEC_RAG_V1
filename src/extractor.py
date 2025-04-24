"""
ChatCompletion function‑calling for dynamic schema extraction.
"""
import openai
import json
from typing import Dict, Any, List
from .config import OPENAI_CHAT_MODEL, FEATURES

FUNCTIONS = [
    {
        "name": "extract_fields",
        "description": "Extract values for each feature defined in FEATURES schema from SEC filing chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "Schema dict for DataFrame features",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "type":        {"type": "string"},
                            "enum": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "required":    {"type": "boolean"}
                        },
                        "required": ["description","type","required"]
                    }
                },
                "records": {
                    "type": "array",
                    "description": "List of chunk metadata + text objects",
                    "items": {"type": "object"}
                }
            },
            "required": ["features","records"]
        }
    }
]

def extract_fields(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Construct messages
    messages = [
        {"role": "system", "content": "You are an assistant that extracts structured data from SEC filings."},
        {"role": "assistant", "content": f"Schema for extraction: {json.dumps(FEATURES)}"},
        {"role": "user", "content": "Extract the following features from the provided SEC filing chunks."}
    ]
    for chunk in chunks:
        messages.append({"role": "assistant", "content": chunk.get("text", "")})

    resp = openai.ChatCompletion.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        functions=FUNCTIONS,
        function_call={"name": "extract_fields"}
    )
    args = resp["choices"][0]["message"]["function_call"]["arguments"]
    result = json.loads(args)
    return result.get("records", [])

