

# logger.py

import json
import time
from datetime import datetime

class RequestLogger:
    def __init__(self, log_file="logs.jsonl"):
        self.log_file = log_file

    def log(self, log_object: dict):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_object) + "\n")

    @staticmethod
    def create_base_log(conversation_id, user_query):
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "conversation_id": conversation_id,
            "user_query": user_query,
        }
