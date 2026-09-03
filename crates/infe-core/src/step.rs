#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
//! Step contract traits — the core abstraction of the infe framework.
//!
//! Every component implements a contract that is called **once per engine
//! step**: the engine gathers its state into [`StepInput`] (arrays + metadata),
//! calls the component, and reads [`StepOutput`] (arrays + plan). There are no
//! per-request or per-token Python calls (BRIEF §5.1).
//!
//! The contract is deliberately engine-agnostic. The shim layer translates
//! between the engine's internal types and these structs.

use crate::buffer::BufferView;
use crate::error::ComponentResult;
use std::collections::BTreeMap;

/// Metadata describing the current scheduling step.
///
/// This is the "scalar" context that accompanies the array inputs. It carries
/// the information a scheduler or KV manager needs to make decisions without
/// touching the actual token data.
#[derive(Debug, Clone, Default)]
pub struct StepContext {
    /// Engine step sequence number (monotonically increasing within a run).
    pub step_id: u64,

    /// Number of requests currently in the engine (running + waiting).
    pub num_requests: usize,

    /// Number of requests actively decoding this step.
    pub num_running: usize,

    /// Number of requests waiting for prefill.
    pub num_waiting: usize,

    /// Maximum batch size the engine allows this step.
    pub max_batch_size: usize,

    /// Maximum number of tokens that can be scheduled for prefill this step.
    pub max_prefill_tokens: usize,

    /// Free KV blocks available for allocation.
    pub free_kv_blocks: usize,

    /// Total KV blocks in the pool.
    pub total_kv_blocks: usize,

    /// Engine configuration flags that affect scheduling, as opaque key-value
    /// pairs. The component reads the ones it understands and ignores the rest.
    pub engine_config: BTreeMap<String, String>,
}

impl StepContext {
    /// Fraction of KV blocks that are free (0.0 = full, 1.0 = empty).
    #[must_use]
    pub fn kv_free_fraction(&self) -> f64 {
        if self.total_kv_blocks == 0 {
            return 1.0;
        }
        self.free_kv_blocks as f64 / self.total_kv_blocks as f64
    }

    /// Whether the engine is under memory pressure (< 10% free KV blocks).
    #[must_use]
    pub fn is_kv_pressured(&self) -> bool {
        self.kv_free_fraction() < 0.1
    }
}

/// A single request in the step input.
///
/// The engine packs its request state into these structs; the component
/// operates on them without touching the engine's internal request objects.
#[derive(Debug, Clone)]
pub struct StepRequest {
    /// Unique request ID (opaque to the component; echoed in the output).
    pub id: u64,

    /// Number of prompt tokens (prefill).
    pub num_prompt_tokens: usize,

    /// Number of tokens already generated (decode).
    pub num_output_tokens: usize,

    /// KV blocks currently held by this request.
    pub kv_blocks_held: usize,

    /// Hash of the prompt prefix (for prefix-cache matching). 0 if none.
    pub prefix_hash: u64,

    /// Priority (0 = normal, higher = more important).
    pub priority: i32,

    /// Whether this request is in the prefill phase.
    pub is_prefill: bool,

    /// Whether this request is using speculative decoding.
    pub is_spec_decode: bool,
}

/// Input arrays for a step, keyed by a stable string name.
///
/// The names are part of the component contract and are documented in the
/// manifest. Example names: `"token_ids"`, `"block_table"`, `"slot_mapping"`,
/// `"cu_seqlens"`.
pub type StepInput = BTreeMap<String, BufferView<'static>>;

/// A scheduling decision for a single request in the step plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepDecision {
    /// Admit this request into the batch for this step.
    Admit {
        /// Number of tokens to process this step (for chunked prefill).
        num_tokens: usize,
        /// KV blocks to allocate for this request.
        num_kv_blocks: usize,
    },

    /// Defer this request to a future step.
    Defer,

    /// Preempt this request (evict from KV cache, return to waiting).
    Preempt {
        /// Whether to keep the prefix cache entry.
        keep_prefix: bool,
    },

    /// Finish this request (no more work needed).
    Finish,
}

/// The step plan: the component's decision for each request.
#[derive(Debug, Clone, Default)]
pub struct StepPlan {
    /// Decisions, keyed by request ID.
    pub decisions: BTreeMap<u64, StepDecision>,

    /// Total KV blocks to allocate this step.
    pub total_kv_blocks_to_allocate: usize,

    /// Total KV blocks freed this step.
    pub total_kv_blocks_freed: usize,

    /// Total tokens scheduled for prefill this step.
    pub total_prefill_tokens: usize,

    /// Total tokens scheduled for decode this step.
    pub total_decode_tokens: usize,
}

impl StepPlan {
    /// Create an empty plan.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a decision for a request.
    pub fn add(&mut self, request_id: u64, decision: StepDecision) {
        match &decision {
            StepDecision::Admit {
                num_tokens,
                num_kv_blocks,
            } => {
                if num_tokens > &0 {
                    // Prefill or decode tokens
                }
                self.total_kv_blocks_to_allocate += num_kv_blocks;
            }
            StepDecision::Preempt { .. } => {
                self.total_kv_blocks_freed += 1; // at least one block freed
            }
            _ => {}
        }
        self.decisions.insert(request_id, decision);
    }

    /// Number of requests admitted this step.
    #[must_use]
    pub fn num_admitted(&self) -> usize {
        self.decisions
            .values()
            .filter(|d| matches!(d, StepDecision::Admit { .. }))
            .count()
    }

    /// Number of requests preempted this step.
    #[must_use]
    pub fn num_preempted(&self) -> usize {
        self.decisions
            .values()
            .filter(|d| matches!(d, StepDecision::Preempt { .. }))
            .count()
    }
}

/// Output from a component step.
///
/// Components that produce scheduling decisions fill in `plan`. Components that
/// produce data (e.g. parsers) fill in `output_buffers`. Most components fill
/// only one of the two.
#[derive(Debug)]
pub struct StepOutput {
    /// Scheduling decisions (for `infe-sched`, `infe-kv`).
    pub plan: Option<StepPlan>,

    /// Output arrays (for `infe-parsers`, attention metadata builders).
    /// Keys are defined by the component contract.
    pub output_buffers: BTreeMap<String, Vec<u8>>,

    /// Whether this step was a no-op (no work done). The engine can use this
    /// to skip processing. Defaults to `true` — an empty output means nothing
    /// was done.
    pub no_op: bool,
}

impl Default for StepOutput {
    fn default() -> Self {
        Self {
            plan: None,
            output_buffers: BTreeMap::new(),
            no_op: true,
        }
    }
}

/// Result of a step call.
pub type StepResult = ComponentResult<StepOutput>;

/// The core trait that every infe component implements.
///
/// One call per engine step. The engine gathers its state into `input` and
/// `context`, calls `step`, and reads the output. Rust releases the GIL for
/// the duration of this call.
///
/// Components are stateful — they maintain internal state across steps (e.g.
/// the KV cache tree, the scheduler's request queue). The engine creates one
/// instance at startup and calls `step` repeatedly.
pub trait StepComponent: Send + Sync {
    /// The component name (e.g. "infe-kv", "infe-parsers").
    const NAME: &'static str;

    /// Process one engine step.
    ///
    /// # Arguments
    ///
    /// * `input` - Array inputs for this step (token ids, block tables, etc.)
    /// * `context` - Scalar metadata for this step (request counts, KV state)
    ///
    /// # Returns
    ///
    /// A [`StepOutput`] containing scheduling decisions and/or output arrays,
    /// or a [`ComponentError`](crate::ComponentError) if the step fails.
    fn step(&mut self, input: &StepInput, context: &StepContext) -> StepResult;

    /// Reset the component to its initial state.
    ///
    /// Called when the engine resets (e.g. on model reload). The component
    /// should clear all internal state.
    fn reset(&mut self) {
        // Default: no-op. Components override if they have state.
    }

    /// Report internal metrics as a flat map of name → value.
    ///
    /// Called periodically by the engine (or by `StepTimer`) for observability.
    /// Example keys: `"cache_hit_rate"`, `"preemptions_total"`,
    /// `"requests_queued"`.
    fn metrics(&self) -> BTreeMap<String, f64> {
        BTreeMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_context_kv_pressure() {
        let ctx = StepContext {
            free_kv_blocks: 20,
            total_kv_blocks: 100,
            ..Default::default()
        };
        assert!(!ctx.is_kv_pressured()); // 20% free is not pressured

        let ctx = StepContext {
            free_kv_blocks: 5,
            total_kv_blocks: 100,
            ..Default::default()
        };
        assert!((ctx.kv_free_fraction() - 0.05).abs() < 1e-9);
        assert!(ctx.is_kv_pressured()); // 5% free IS pressured

        let ctx = StepContext {
            free_kv_blocks: 9,
            total_kv_blocks: 100,
            ..Default::default()
        };
        assert!(ctx.is_kv_pressured());

        let ctx = StepContext {
            free_kv_blocks: 10,
            total_kv_blocks: 100,
            ..Default::default()
        };
        assert!(!ctx.is_kv_pressured()); // exactly 10% is not pressured
    }

    #[test]
    fn step_plan_admit_and_preempt() {
        let mut plan = StepPlan::new();
        plan.add(
            1,
            StepDecision::Admit {
                num_tokens: 128,
                num_kv_blocks: 4,
            },
        );
        plan.add(2, StepDecision::Defer);
        plan.add(3, StepDecision::Preempt { keep_prefix: true });
        plan.add(4, StepDecision::Finish);

        assert_eq!(plan.num_admitted(), 1);
        assert_eq!(plan.num_preempted(), 1);
        assert_eq!(plan.total_kv_blocks_to_allocate, 4);
        assert_eq!(plan.decisions.len(), 4);
    }

    #[test]
    fn step_output_no_op_default() {
        let out = StepOutput::default();
        assert!(out.no_op);
        assert!(out.plan.is_none());
    }

    #[test]
    fn step_context_empty_kv() {
        let ctx = StepContext::default();
        // No KV blocks → fully free (vacuously)
        assert!((ctx.kv_free_fraction() - 1.0).abs() < 1e-9);
        assert!(!ctx.is_kv_pressured());
    }
}
