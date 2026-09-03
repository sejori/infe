# infe-core

Core infrastructure for the **infe** component framework: step-granular contract
traits, buffer helpers for PyO3 crossings, unified error types, and a
`StepTimer` for CPU-side step instrumentation.

## What this crate provides

| Module | What |
|---|---|
| `step` | The `StepComponent` trait, `StepInput`/`StepOutput`, `StepContext`, `StepPlan`, `StepDecision` |
| `buffer` | `BufferView` / `BufferViewMut` — typed views over DLPack/numpy buffers that cross the PyO3 boundary without copying |
| `error` | `ComponentError` / `ComponentResult` — structured errors the shim maps to engine exceptions |
| `timer` | `StepTimer` — near-zero-overhead CPU step timer with min/max/mean/last stats |
| `manifest` | `ComponentManifest` — the v0 manifest schema for the registry |

## The boundary rule

Every component contract is **one call per engine step**: arrays in, arrays
out. Rust releases the GIL for the duration of the call. Data crosses as
DLPack / Arrow / numpy views over pre-allocated buffers, never as Python
objects. See `BRIEF.md` §5.1.

## Contract version

0 (this is M0; the contract will stabilise before M1 ships).

## What this replaces

Nothing yet — `infe-core` is the foundation. The first component (`infe-parsers`)
will replace both engines' tool-call and reasoning parser suites.

## Last A/B result

Not yet benchmarked (M0).
