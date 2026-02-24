# Project Overview

## What This Is

`sugar.model` is the machine learning component of a larger system that brings AI-assisted reflection to the Sugar Learning Platform. When a child finishes or pauses an activity in Sugar, the system generates a thoughtful, age-appropriate question that encourages them to reflect on what they did and what they learned.

## The Broader Project

This repository is one part of a GSoC 2026 project titled **"AI Reflection in the Sugar Journal"**. The full project spans multiple repositories:

```
gsoc2026/
├── idea.md                  The original GSoC project proposal
├── sugar/                   Fork of the Sugar desktop shell (GTK3/Python)
│   └── src/jarabe/model/
│       └── reflection.py    Client-side reflection service (talks to backend)
├── architect/
│   ├── ARCHITECTURE.md      Full system architecture document
│   └── ai-reflection-service/   FastAPI backend that serves reflection prompts
├── sugar.model/             This repository — model training and inference
├── sugar.test/              Testing activity for Sugar
└── test_reflection_standalone.py   Standalone test for the reflection service
```

## How It All Fits Together

```
Child closes activity
        │
        ▼
Sugar Shell (reflection.py)
        │  Sends activity context via HTTP POST
        ▼
FastAPI Backend (architect/ai-reflection-service/)
        │  Routes to framework, builds prompt, calls model
        ▼
sugar.model (this repo)
        │  Generates a reflection question
        ▼
Response flows back to Sugar Shell
        │
        ▼
Reflection dialog appears for the child
```

The Sugar Shell client sends activity metadata (title, type, duration, age) to the FastAPI backend. The backend selects an appropriate reflection framework, constructs a prompt, and calls the fine-tuned model. The model returns a reflection question, which is displayed to the child in a dialog window.

## What the Model Does

It takes structured activity context and produces a single reflection question.

**Input:** `Activity: TurtleArt Activity | Duration: 15 min | Age: 9 | Framework: what_so_what | Stage: what`

**Output:** `"You just made something in TurtleArt! What was the most interesting part of your project?"`

The model does not select frameworks, route requests, or manage conversations. It generates one question from one context. Everything else is handled by the backend and the Sugar Shell.
