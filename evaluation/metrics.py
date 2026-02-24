# -*- coding: utf-8 -*-
"""
Quality metrics for evaluating generated reflection prompts.
"""

import re


def is_question(text):
    """Check whether the text contains a question mark."""
    return "?" in text


def is_open_ended(text):
    """Determine if the question is open-ended based on its opening word."""
    openers = [
        "what", "how", "why", "when", "where", "who",
        "tell", "describe", "explain", "walk", "think",
        "look", "compare", "can you", "could you",
    ]
    text_lower = text.lower().strip()
    for opener in openers:
        if text_lower.startswith(opener):
            return True
    return False


def references_activity(text, title):
    """Check whether the prompt mentions the activity title."""
    return title.lower() in text.lower()


def appropriate_length(text, min_words=5, max_words=60):
    """Check whether the prompt falls within an acceptable word count."""
    word_count = len(text.split())
    return min_words <= word_count <= max_words


def age_appropriate_vocabulary(text, age):
    """Flag overly complex vocabulary for younger learners."""
    if age is None or age > 12:
        return True

    complex_words = [
        "conceptualize", "methodology", "furthermore", "synthesize",
        "subsequently", "articulate", "paradigm", "hypothesis",
        "juxtapose", "extrapolate", "metacognitive",
    ]

    text_lower = text.lower()
    for word in complex_words:
        if word in text_lower:
            return False

    if age < 8:
        words = text.split()
        avg_len = sum(len(w) for w in words) / max(len(words), 1)
        if avg_len > 7:
            return False

    return True


def score_single(generated, reference=None, title="", age=None):
    """Score a single generated prompt across all metrics."""
    scores = {
        "is_question": int(is_question(generated)),
        "is_open_ended": int(is_open_ended(generated)),
        "appropriate_length": int(appropriate_length(generated)),
        "age_appropriate": int(age_appropriate_vocabulary(generated, age)),
    }

    if title:
        scores["references_activity"] = int(
            references_activity(generated, title)
        )

    scores["total"] = sum(scores.values())
    scores["max_total"] = len(scores) - 1

    return scores


def score_batch(results):
    """Score a batch of results and return per-example scores with an aggregate summary."""
    all_scores = []
    for r in results:
        s = score_single(
            generated=r["generated"],
            reference=r.get("reference"),
            title=r.get("title", ""),
            age=r.get("age"),
        )
        all_scores.append(s)

    if not all_scores:
        return all_scores, {}

    metric_names = [k for k in all_scores[0] if k not in ("total", "max_total")]
    summary = {}
    for m in metric_names:
        values = [s[m] for s in all_scores]
        summary[m] = sum(values) / len(values)

    summary["avg_total"] = sum(s["total"] for s in all_scores) / len(all_scores)
    summary["count"] = len(all_scores)

    return all_scores, summary
