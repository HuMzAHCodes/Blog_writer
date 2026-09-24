# Lab Report — Orchestrator-Worker Agent (Dynamic Parallel Fan-Out)

## Topic

A blog-writing agent built with LangGraph, using the **orchestrator-worker
pattern** (also called map-reduce for agents): one LLM call plans the work
into a variable number of subtasks, each subtask is dispatched to a parallel
"worker" LLM call, and the results are merged into one final document and
saved to disk.

## Coding Techniques and Features Used

**Structured planning via `with_structured_output`.**
```python
class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="What to cover")

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

plan = llm.with_structured_output(Plan).invoke([...])
```
The orchestrator doesn't just generate a plan as free text — it returns a
validated `Plan` object containing a list of typed `Task`s, guaranteeing
every downstream step (the fan-out, each worker's prompt) can rely on a known
shape rather than parsing an LLM's raw output.

**Dynamic parallel fan-out with `Send`.**
```python
def fanout(state: State):
    return [Send("worker", {"task": task, "topic": state["topic"], "plan": state["plan"]})
            for task in state["plan"].tasks]

g.add_conditional_edges("orchestrator", fanout, ["worker"])
```
This is the key mechanic distinguishing this lab from earlier fan-out/fan-in
graphs in this series (e.g. the UPSC essay evaluator, which had exactly three
fixed parallel branches). `Send` lets the **number of parallel branches be
decided at runtime**, based on however many tasks the orchestrator's plan
actually produces — 5 tasks means 5 worker invocations, 7 tasks means 7,
with no code change required. Each `Send` also carries its own custom
payload, so every worker invocation gets exactly the task, topic, and plan
it needs.

**A reducer field for collecting dynamic-count parallel results.**
```python
class State(TypedDict):
    sections: Annotated[List[str], operator.add]
```
Every worker returns `{"sections": [section_md]}` — a one-item list.
`operator.add` concatenates each worker's contribution instead of the last
one overwriting the rest, the same reducer mechanic from the earlier
fan-out/fan-in lab, now working with a dynamic number of writers rather than
a fixed three.

**A dedicated reducer *node*, separate from the reducer *field*.** Worth
noting the naming overlap: `operator.add` is the state-level reducer
function; `reducer` (the node) is a separate, ordinary graph node that runs
*after* all workers finish, merging their sections into one document string
and writing it to disk:
```python
def reducer(state: State) -> dict:
    body = "\n\n".join(state["sections"]).strip()
    final_md = f"# {title}\n\n{body}\n"
    Path(filename).write_text(final_md, encoding="utf-8")
    return {"final": final_md}
```

**Real file output as a side effect.** Unlike prior labs where the graph's
result was only ever printed or returned in memory, this graph's final node
writes an actual `.md` file to disk — the agent produces a tangible
deliverable (a formatted blog post file), not just a returned string.

## Graph Shape

```
START → orchestrator → [dynamic fan-out via Send] → worker (×N, parallel) → reducer → END
```

The `worker` node has exactly one incoming and one outgoing edge in the graph
definition, yet executes N times per run — the multiplicity comes entirely
from `Send`, not from the static edge structure.

## Conclusion

This lab introduces two genuinely new LangGraph capabilities beyond earlier
parallel-graph work: `with_structured_output` driving a **planning** step
whose output determines the graph's own execution shape, and `Send` enabling
**dynamic-count** parallel fan-out rather than a fixed number of hardcoded
branches. Together they form a general pattern — plan, dispatch variably-many
parallel workers, merge — applicable well beyond blog writing to any task
that decomposes into an LLM-determined number of independent sub-jobs.
