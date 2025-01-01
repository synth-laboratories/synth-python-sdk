from pydantic import BaseModel
from typing import List, Dict, Optional, Literal, Union
import json

class OpenAPISpec(BaseModel):
    openapi: str = "3.1.0"
    info: Dict = {
        "title": "Synth SDK",
        "version": "0.2.74",
        "description": "SDK for tracing and evaluating AI system behavior"
    }
    servers: List[Dict] = [{"url": "https://api.usesynth.ai"}]
    paths: Dict = {
        "/trace": {
            "post": {
                "summary": "Trace system execution",
                "description": "Decorator for tracing method execution in an AI system",
                "parameters": [
                    {
                        "name": "origin",
                        "in": "query",
                        "required": True,
                        "schema": {
                            "type": "string",
                            "enum": ["agent", "environment"]
                        },
                        "description": "Source of the computation (AI/model operations or external system operations)"
                    },
                    {
                        "name": "event_type",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Type of event being traced (e.g., inference, training)"
                    },
                    {
                        "name": "manage_event",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["create", "end", None]
                        },
                        "description": "Controls the lifecycle of the event"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Successfully traced execution"
                    }
                }
            }
        },
        "/track/lm": {
            "post": {
                "summary": "Track language model interaction",
                "parameters": [
                    {
                        "name": "messages",
                        "in": "body",
                        "required": True,
                        "schema": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"}
                                }
                            }
                        }
                    },
                    {
                        "name": "model_name",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ]
            }
        },
        "/track/state": {
            "post": {
                "summary": "Track state changes",
                "parameters": [
                    {
                        "name": "variable_name",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"}
                    },
                    {
                        "name": "variable_value",
                        "in": "body",
                        "required": True,
                        "schema": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "object"},
                                {"type": "array"}
                            ]
                        }
                    },
                    {
                        "name": "origin",
                        "in": "query",
                        "required": True,
                        "schema": {
                            "type": "string",
                            "enum": ["agent", "environment"]
                        }
                    }
                ]
            }
        }
    }

def generate_openapi():
    spec = OpenAPISpec()
    
    # Save to file
    with open("openapi.json", "w") as f:
        json.dump(spec.model_dump(), f, indent=2)

if __name__ == "__main__":
    generate_openapi()
