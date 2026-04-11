"""
Conversational Memory
=====================
Stores chat history for multi-turn Q&A so students
can ask follow-up questions without repeating context.
"""

from collections import deque


class ConversationMemory:
    """Maintains a sliding window of recent exchanges."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)

    def add(self, question: str, answer: str):
        """Store a Q&A exchange."""
        self.history.append({"question": question, "answer": answer})

    def get_context_string(self) -> str:
        """Format history as a string for the LLM prompt."""
        if not self.history:
            return ""

        lines = []
        for turn in self.history:
            lines.append(f"Student: {turn['question']}")
            # Truncate long answers to keep prompt manageable
            short_answer = turn["answer"][:300]
            if len(turn["answer"]) > 300:
                short_answer += "..."
            lines.append(f"Nova: {short_answer}")

        return "\n".join(lines)

    def get_follow_up_query(self, new_question: str) -> str:
        """
        Rewrite a follow-up question to be self-contained using chat history.
        E.g., "What about L1?" after discussing regularization
        becomes "What is L1 regularization?"
        """
        if not self.history:
            return new_question

        # If the question seems self-contained already, return as-is
        standalone_signals = ["what is", "explain", "how does", "define", "describe"]
        q_lower = new_question.lower().strip()
        if len(q_lower.split()) > 8 or any(q_lower.startswith(s) for s in standalone_signals):
            return new_question

        # Build context from last exchange for query rewriting
        last = self.history[-1]
        return f"{last['question']} {new_question}"

    def clear(self):
        """Reset conversation history."""
        self.history.clear()

    @property
    def turn_count(self) -> int:
        return len(self.history)