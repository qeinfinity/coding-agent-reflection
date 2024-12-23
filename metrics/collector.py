# metrics/collector.py
import sqlite3
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
import tiktoken

class MetricsCollector:
    def __init__(self, model_name: str = "gpt-4o", db_path: str = "metrics.db"):
        """Initialize metrics collector with SQLite storage."""
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.db_path = db_path
        self.session_id = datetime.now().isoformat()
        self.current_interaction = None
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with necessary tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT,
                    timestamp TEXT,
                    prompt TEXT,
                    solution TEXT,
                    reflection TEXT,
                    total_tokens INTEGER,
                    response_time REAL,
                    layers_accessed TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS completions (
                    id INTEGER PRIMARY KEY,
                    interaction_id INTEGER,
                    context_type TEXT,
                    duration REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    FOREIGN KEY(interaction_id) REFERENCES interactions(id)
                )
            """)

    def start_interaction(self):
        """Start a new interaction."""
        self.current_interaction = {
            'start_time': time.time(),
            'completions': [],
            'layer_access': set(),
        }

    def log_layer_access(self, layer: int):
        """Log access to a specific memory layer."""
        if self.current_interaction:
            self.current_interaction['layer_access'].add(layer)

    def log_completion(self, context_type: str, duration: float, prompt_tokens: int, completion_tokens: int):
        """Log a completion from the OpenAI API."""
        if self.current_interaction:
            self.current_interaction['completions'].append({
                'context_type': context_type,
                'duration': duration,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            })

    def end_interaction(self, prompt: str, solution: str, reflection: str, layers_accessed: list):
        """End the current interaction and save metrics."""
        if not self.current_interaction:
            return

        end_time = time.time()
        duration = end_time - self.current_interaction['start_time']
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert interaction
            cursor = conn.execute(
                """
                INSERT INTO interactions 
                (session_id, timestamp, prompt, solution, reflection, total_tokens, 
                 response_time, layers_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    datetime.now().isoformat(),
                    prompt,
                    solution,
                    reflection,
                    self._count_tokens(prompt + solution + reflection),
                    duration,
                    json.dumps(list(self.current_interaction['layer_access']))
                )
            )
            
            interaction_id = cursor.lastrowid
            
            # Insert completions
            for completion in self.current_interaction['completions']:
                conn.execute(
                    """
                    INSERT INTO completions 
                    (interaction_id, context_type, duration, prompt_tokens, completion_tokens)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        interaction_id,
                        completion['context_type'],
                        completion['duration'],
                        completion['prompt_tokens'],
                        completion['completion_tokens']
                    )
                )

        self.current_interaction = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        return len(self.encoding.encode(text))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics for the current session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get session metrics
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as interaction_count,
                    AVG(total_tokens) as avg_tokens,
                    AVG(response_time) as avg_response_time
                FROM interactions
                WHERE session_id = ?
                """,
                (self.session_id,)
            )
            
            summary = dict(cursor.fetchone())
            
            # Get layer access patterns
            cursor = conn.execute(
                """
                SELECT layers_accessed
                FROM interactions
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
                """,
                (self.session_id,)
            )
            
            layer_patterns = [json.loads(row['layers_accessed']) for row in cursor]
            summary['recent_layer_patterns'] = layer_patterns
            
            return summary