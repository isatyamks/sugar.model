# Reflection Frameworks

## Background

Reflective practice is a structured approach to thinking about experiences in order to learn from them. In the context of Sugar, we want children to pause and think about what they did, why they did it, and what they might do differently next time.

The model supports three established pedagogical frameworks, each suited to different ages and types of thinking.

## What? So What? Now What?

Developed by Borton (1970) and refined by Driscoll (2007). The simplest of the three frameworks, designed for young learners who may be new to reflection.

**Stages:**
- **What?** — Describe what happened. *"What did you make?"*
- **So What?** — Why does it matter? *"What did you learn from that?"*
- **Now What?** — What comes next? *"What would you try differently next time?"*

**Used for:** Learners under 12. Questions should be short (one sentence), warm, and use simple words.

## Gibbs Reflective Cycle

Introduced by Gibbs (1988). A six-stage cycle that encourages deeper, more structured reflection. Appropriate for older learners who can articulate feelings and analyze their own thinking.

**Stages:**
- **Description** — What happened? (facts only)
- **Feelings** — What were you feeling?
- **Evaluation** — What went well or badly?
- **Analysis** — Why did it happen that way?
- **Conclusion** — What did you learn?
- **Action Plan** — What will you do next time?

**Used for:** Learners 12 and older. Questions can be longer (1–2 sentences) and expect more thoughtful responses.

## Kolb Experiential Learning Cycle

Proposed by Kolb (1984). Emphasizes learning through doing, observing, thinking, and experimenting. Well suited for creative and hands-on activities like TurtleArt, Pippy, and Music Blocks.

**Stages:**
- **Concrete Experience** — What exactly did you do?
- **Reflective Observation** — What stands out when you look at your work?
- **Abstract Conceptualization** — What patterns or ideas did you notice?
- **Active Experimentation** — How could you test that idea next time?

**Used for:** Learners ages 10–14, especially with coding and creative activities.

## Framework Selection

The model itself does not choose which framework to use. Framework selection is handled by the backend routing logic in `architect/ai-reflection-service/`. The router picks a framework based on the learner's age and the type of activity, then passes it to the model as part of the input context.

In the training data, framework assignment follows similar age-based rules:
- Under 10 → what_so_what
- 10 to 12 → what_so_what or kolb
- 12 and older → gibbs or kolb
