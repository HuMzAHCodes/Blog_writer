# Lab Report — Blog Writer Agent, v3 → v4: Research Routing, Then Recency Hardening

## Topic

Two files covering the same major architectural leap: adding **conditional
web research** to the orchestrator-worker pipeline via a new routing node,
real search (Tavily), and evidence-grounded citations. v3 introduces this
capability; v4 hardens it with date-awareness and closes a couple of
correctness gaps v3 left open.

## v2 → v3: The Structural Leap (Research Routing)

Unlike v1→v2 (schema/prompt tuning only, identical graph), this jump is a
genuine architecture change.

### New node: the router, deciding research need up front
```python
class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"
```
Before any planning happens, an LLM call classifies the topic into one of
three modes — `closed_book` (evergreen, no research needed), `hybrid` (mostly
evergreen but benefits from current examples), or `open_book` (inherently
time-sensitive, e.g. weekly roundups). This decision conditionally routes the
graph: `closed_book` skips straight to the orchestrator; the other two modes
detour through a new `research` node first.

### New node: real web search via Tavily, synthesized into structured evidence
```python
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    ...

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None
```
Raw search results (unstructured dicts from Tavily) are passed through
`with_structured_output(EvidencePack)` to normalize them into typed
`EvidenceItem`s, then deduplicated by URL. This is the first file in the
series where the agent's knowledge extends beyond the LLM's training data —
a genuine retrieval step, not just better prompting of what the model
already "knows."

### The orchestrator and worker now consume evidence conditionally
The orchestrator's prompt branches by mode: `closed_book` plans without
evidence at all; `hybrid` uses evidence for supporting examples and flags
which sections need it (`requires_research`, `requires_citations`);
`open_book` forces the entire plan into a `"news_roundup"` structure and
explicitly forbids drifting into tutorial content. The worker enforces a
**grounding policy**: for `open_book` mode, any specific event/company/model
claim must carry a Markdown citation link back to a supplied Evidence URL, or
be replaced with "Not found in provided sources" — preventing the model from
fabricating claims it has no real source for.

### A correctness fix hiding inside this file: ordered parallel output
```python
sections: Annotated[List[tuple[int, str]], operator.add]
...
return {"sections": [(task.id, section_md)]}
...
ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
```
v1/v2 collected worker output as `List[str]` via `operator.add` — but since
workers run in parallel via `Send`, nothing guaranteed they'd *finish* in
dispatch order. v3 fixes this silently-present bug by having each worker
return `(task_id, section_md)` and having the reducer explicitly re-sort by
`task_id` before joining, guaranteeing correct reading order regardless of
which parallel call happens to finish first.

## v3 → v4: Hardening the Research Pipeline

v4 keeps v3's entire architecture intact and adds precision plus two
robustness fixes.

### 1. Date-awareness: `as_of` and computed `recency_days`
```python
class State(TypedDict):
    ...
    as_of: str
    recency_days: int
```
```python
if decision.mode == "open_book":
    recency_days = 7
elif decision.mode == "hybrid":
    recency_days = 45
else:
    recency_days = 3650
```
The router now also determines *how recent* evidence needs to be, based on
mode — a weekly roundup needs sources from the last 7 days; a hybrid
explainer tolerates 45; a closed-book topic effectively has no recency
requirement at all (a ~10-year window).

### 2. A hard recency filter, not just a prompt instruction
```python
if mode == "open_book":
    as_of = date.fromisoformat(state["as_of"])
    cutoff = as_of - timedelta(days=int(state["recency_days"]))
    fresh = [e for e in evidence if _iso_to_date(e.published_at) and _iso_to_date(e.published_at) >= cutoff]
    evidence = fresh
```
This is the most significant upgrade: rather than *asking* the model to only
use recent sources (which it might not reliably follow), v4 actually
**deletes stale evidence in code** before it ever reaches the orchestrator or
workers, for open_book mode. An LLM instruction is a request; a Python date
comparison is a guarantee.

### 3. Two robustness hardenings, closing gaps v3 left implicit
```python
forced_kind = "news_roundup" if mode == "open_book" else None
...
if forced_kind:
    plan.blog_kind = "news_roundup"
```
v3 only *asked* the model to set `blog_kind = "news_roundup"` for open_book
topics via prompt instruction — nothing stopped the model from forgetting.
v4 forces it in code after the structured output returns, regardless of what
the model actually produced.

```python
def fanout(state: State):
    assert state["plan"] is not None
    ...

def reducer_node(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("Reducer called without a plan.")
```
Explicit assertions/checks that a plan exists before continuing — turning a
potential silent `AttributeError` deep in worker/reducer logic into an
immediate, clear failure at the point something actually went wrong.

### 4. A diagnostic runner
v4's `run()` now prints a structured summary after every invocation — mode,
recency window, evidence count and a sample, task count, output size — making
it possible to verify the router's and research node's decisions without
digging through the graph's internal state manually.

## A Real-World Debugging Note (This Account's Model Tier)

Running v3 with `ministral-8b-latest` produced a `ValidationError`: across 9
generated tasks, the model consistently omitted the required `target_words`
field, and one task's `bullets` list fell below the required minimum length.
This was a **structured-output reliability limit of the smaller model** under
a genuinely complex, multi-field, repeated schema — not a bug in the graph or
prompts. Switching to `ministral-14b-latest` resolved it immediately, and v4
adopted that model tier as its default going forward. This mirrors the
project's standing lesson (documented in `MISTRAL_MODEL_LIMITS.md`): schema
complexity and model capability have to be matched, and the fix for a
structured-output failure is often a model-tier change, not a prompt or code
change.

## Evolution Summary

```
DIMENSION                v2                v3                              v4
------------------------ ----------------- ------------------------------- --------------------------------
Graph nodes               3 (orch/worker/  5 (+ router, research)          5 (unchanged)
                           reducer)
Research capability        None             Tavily search + evidence pack   Same, now recency-filtered
Mode awareness              None             closed/hybrid/open_book        Same, now with recency window
Recency handling            None             Prompt instruction only        HARD FILTER in code (as_of/cutoff)
Section ordering            Implicit list    Explicit (task_id, md) tuple    Unchanged
Robustness                  None             None                            Forced blog_kind, plan assertions
Model tier                  ministral-8b     ministral-8b (broke on scale)   ministral-14b (fixed)
```

## Conclusion

v2→v3 is a genuine architectural expansion — a routing decision and a real
retrieval step turn the agent from "write from what the model already knows"
into "decide whether current information is needed, fetch it, and ground
claims in it." v3→v4 doesn't change that architecture at all; it closes the
gap between *asking* the model to behave correctly (via prompt instructions)
and *guaranteeing* it in code — the recency filter, the forced `blog_kind`,
and the plan-existence assertions are all instances of the same principle:
where correctness actually matters, enforce it deterministically rather than
trusting the model to follow an instruction every time.
