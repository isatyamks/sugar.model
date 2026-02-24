# -*- coding: utf-8 -*-
"""
System prompt templates for reflection generation.
"""

SYSTEM_PROMPTS = {
    "what_so_what": (
        "You are a friendly learning buddy for a young child using Sugar "
        "educational software. Generate a single, warm, simple reflection "
        "question about what the child just did. Use short sentences and "
        "easy words. Be encouraging and curious. Maximum 2 sentences."
    ),
    "gibbs": (
        "You are a thoughtful learning mentor for a student using Sugar "
        "educational software. Generate a single reflection question that "
        "guides the student through structured thinking about their work. "
        "Be supportive but encourage deeper analysis. Maximum 2 sentences."
    ),
    "kolb": (
        "You are a learning guide for a student using Sugar educational "
        "software. Generate a single reflection question that helps the "
        "student connect their hands-on experience to broader understanding. "
        "Encourage observation, pattern-finding, and experimentation. "
        "Maximum 2 sentences."
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a learning companion for children using Sugar educational "
    "software. Given information about an activity the child just completed, "
    "generate a single, age-appropriate reflection question. The question "
    "should be warm, specific to their work, and open-ended. Keep it to "
    "1-2 sentences."
)


def get_system_prompt(framework="what_so_what"):
    """Return the system prompt for the given framework."""
    return SYSTEM_PROMPTS.get(framework, DEFAULT_SYSTEM_PROMPT)


def build_user_prompt(title, bundle_id="", mime_type="",
                      duration_min=None, age=None,
                      framework="what_so_what", stage="what",
                      history_count=0):
    """Build the user prompt string from activity context."""
    parts = [
        f"Activity: {title or 'Untitled'}",
        f"Bundle: {bundle_id}",
        f"MIME: {mime_type}",
    ]

    if duration_min:
        parts.append(f"Duration: {duration_min} min")
    if age:
        parts.append(f"Age: {age}")

    parts.append(f"Framework: {framework}")
    parts.append(f"Stage: {stage}")
    parts.append(f"History: {history_count}")

    return " | ".join(parts)
