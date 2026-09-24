# Lab Report — Blog Writer Agent, v1 → v2: Prompting as the Efficiency Lever

## Topic

Same orchestrator-worker graph as v1 (plan → dynamic parallel fan-out →
write sections → merge), unchanged in structure. The upgrade in this file is
entirely in the **schema and prompt design**: a stricter `Task` schema and
far more directive system prompts, aimed at producing genuinely technical,
specific output instead of generic filler.

## What Stayed Exactly the Same

- The graph shape: `START → orchestrator → Send-based fan-out → worker (×N) → reducer → END`.
- `operator.add` reducer on `sections` for collecting parallel worker output.
- `with_structured_output(Plan)` driving the orchestrator.
- Saving the final merged document to a `.md` file on disk.

None of the LangGraph mechanics changed at all — this file proves that
structural sophistication and output quality are two separate levers.

## What Changed, and Why It Matters

### 1. `Task` schema went from loose to constrained

**v1:**
```python
class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="What to cover")
```

**v2:**
```python
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should be able to do/understand after this section.")
    bullets: List[str] = Field(..., min_length=3, max_length=5, description="3–5 concrete, non-overlapping subpoints...")
    target_words: int = Field(..., description="Target word count for this section (120–450).")
    section_type: Literal["intro", "core", "examples", "checklist", "common_mistakes", "conclusion"] = Field(
        ..., description="Use 'common_mistakes' exactly once in the plan."
    )
```

A single free-text `brief` string became five distinct, individually
constrained fields. `min_length`/`max_length` on `bullets` forces a specific
range of subpoints per section — not zero, not fifteen. The `Literal`
`section_type` with an explicit "exactly once" instruction gives the plan a
*documented structure* (an intro, several core sections, one common-mistakes
section, a conclusion) that v1 had no way to guarantee.

**Why this matters:** in v1, the orchestrator could produce sections with
wildly inconsistent scope and depth, since "brief" gave the model almost
total freedom. v2's schema is itself a specification the model must satisfy —
the schema is doing enforcement work that used to depend entirely on the
model's own judgment.

### 2. `Plan` schema gained explicit audience and tone

**v1:** `blog_title`, `tasks` only.
**v2:** adds `audience` and `tone` fields.

This information then flows into every worker's prompt (`f"Audience:
{plan.audience}\nTone: {plan.tone}"`), so every section is written
consistently for the *same* intended reader and voice — something v1 had no
mechanism to enforce, since nothing in its `Plan` captured who the blog was
even for.

### 3. Orchestrator's prompt: one sentence → a full specification

**v1's entire instruction:** *"Create a blog plan with 5-7 sections on the
following topic."*

**v2's instruction** adds, among other things: an exact structural template
(problem → intuition → approach → implementation → trade-offs →
testing/observability → conclusion), a rule that bullets must be
"actionable and testable" with concrete examples of what that means, a
requirement that the plan include at least one of five specific technical
elements (a minimal working example, edge cases, performance/cost
considerations, security considerations, or observability tips), and an
explicit anti-pattern warning ("avoid vague bullets like 'Explain X' or
'Discuss Y'").

**Why this matters:** v1's short prompt left the model to infer what "good"
looks like. v2 states it directly, including what to avoid — this is the
difference between hoping the model produces something specific and
constraining it to.

### 4. Worker's prompt: minimal formatting instruction → a full technical quality bar

**v1's worker prompt:** *"Write one clean Markdown section."*

**v2's worker prompt** adds a hard word-count tolerance (±15% of
`target_words`), a requirement to cover *all* bullets in order without
skipping or merging them, a technical quality bar (prefer concrete APIs,
data structures, protocols over abstractions), a requirement to include at
least one of a code snippet / example input-output / checklist / text
diagram, an instruction to explain trade-offs and edge cases, and specific
Markdown style rules (start with `## <Section Title>`, use code fences, avoid
marketing language).

**Why this matters:** v1 trusted "clean Markdown" to imply good technical
writing. v2 makes the actual bar for "good" explicit and checkable — a human
reviewer (or the model itself, if this were extended with a review step)
could verify most of these constraints directly against the output.

## Evolution Summary

```
DIMENSION                v1                          v2
------------------------ --------------------------- --------------------------------------
Task schema fields        2 (title, brief)            5 (goal, bullets, target_words, type)
Plan schema fields        2 (title, tasks)             4 (+ audience, tone)
Orchestrator prompt       1 sentence                   ~20 lines, explicit structure + anti-patterns
Worker prompt             1 sentence                   ~15 lines, quality bar + format rules
Graph structure           orchestrator→fanout→worker→reducer   UNCHANGED
```

## Conclusion

The jump from v1 to v2 demonstrates that "improving an agent" doesn't always
mean adding more nodes, more tools, or more graph complexity — it can mean
making the **contract** between the orchestrator and the worker, and between
the system and the model, dramatically more explicit. A stricter Pydantic
schema and a more directive prompt cost nothing architecturally (the graph is
identical) but meaningfully constrain what the model is allowed to produce,
trading a small amount of prompt-writing effort for a large reduction in
the model's freedom to produce vague or generic content. This is the first
of several efficiency upgrades in this series; later files will likely
introduce actual structural changes (validation loops, retries, parallel
review steps) on top of this same orchestrator-worker foundation.
