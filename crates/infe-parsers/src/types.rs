//! Delta types emitted by the streaming parser.
//!
//! These mirror the `OpenAI` streaming chunk format so they can be serialised
//! directly into the SSE stream by the API server with zero translation.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

/// A structured tool-call delta, emitted as the parser discovers tool-call
/// content in the token stream.
///
/// Each delta carries a *fragment* of the arguments JSON string — the API
/// server appends it to the accumulated arguments string (matching `OpenAI`'s
/// streaming protocol).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Index of the tool call in the batch (0-based, matching `OpenAI`'s
    /// streaming `index` field). Auto-incremented per tool call.
    pub index: usize,

    /// Optional tool-call id (present on the first delta for this call).
    /// Generated automatically when a new tool call is detected.
    pub id: Option<String>,

    /// Optional function name (present on the first delta, extracted
    /// from the dialect's name field — e.g. Hermes `{"name": …}`).
    pub name: Option<String>,

    /// Fragment of the arguments JSON string. This is the *diff* — only
    /// characters not yet emitted in a previous delta. The consumer
    /// accumulates these to build the full arguments string.
    pub arguments_fragment: String,

    /// Whether this delta completes the tool call (closing delimiter seen).
    pub is_complete: bool,
}

/// A reasoning-content delta — the `reasoning_content` field in the streaming
/// response, used by thinking models (DeepSeek-R1, Qwen-QwQ, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningDelta {
    /// Fragment of reasoning text.
    pub fragment: String,

    /// Whether this delta completes the reasoning block.
    pub is_complete: bool,
}

/// Internal state machine state for a single tool call being parsed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolCallState {
    /// Not yet inside a tool-call block.
    #[default]
    Idle,
    /// Inside the tool-call block, accumulating JSON.
    InToolCall,
    /// Tool call is complete (closing delimiter seen).
    Complete,
}

/// Internal state for reasoning content.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReasoningState {
    #[default]
    Idle,
    InReasoning,
    Complete,
}

/// Internal state for a single tool-call argument parser (per-request).
///
/// Tracks the current tool call being parsed, including auto-incremented
/// index, generated id, and the amount of arguments already emitted as
/// diffs (to avoid re-sending content on subsequent deltas).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToolCallArgState {
    /// Current parser state.
    pub state: ToolCallState,

    /// Accumulated inner JSON string (between the open/close markers).
    pub arguments_buffer: String,

    /// Tool-call id, assigned when the call is first detected.
    pub id: Option<String>,

    /// Index assigned when this tool call was detected.
    pub index: Option<usize>,

    /// Counter for the next tool call index (auto-incremented).
    pub next_index: usize,

    /// How many characters of the arguments value have already been emitted
    /// in previous deltas. Used to compute diffs so we never re-send content.
    pub last_emitted_args_len: usize,

    /// Whether the name has been emitted in a delta yet (name goes on the
    /// first delta only, matching `OpenAI`'s streaming protocol).
    pub name_emitted: bool,
}

impl ToolCallArgState {
    /// Create a new idle state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a new tool call: assign index, generate id, reset per-call fields.
    pub fn begin_tool_call(&mut self) {
        self.state = ToolCallState::InToolCall;
        self.index = Some(self.next_index);
        self.next_index += 1;
        self.id = Some(make_tool_call_id());
        self.arguments_buffer.clear();
        self.last_emitted_args_len = 0;
        self.name_emitted = false;
    }

    /// Complete the current tool call and return to idle.
    pub fn complete_tool_call(&mut self) {
        self.state = ToolCallState::Complete;
    }

    /// Transition from Complete back to Idle for the next tool call.
    pub fn reset_to_idle(&mut self) {
        if self.state == ToolCallState::Complete {
            self.state = ToolCallState::Idle;
        }
    }
}

// ---------------------------------------------------------------------------
// Tool-call ID generation (B7)
// ---------------------------------------------------------------------------

/// Global monotonic counter for ID uniqueness across requests.
static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Per-thread state for fast pseudo-random mixing initialised from the
    /// global counter and an address (ASLR entropy).
    static RNG_STATE: Cell<u64> = const { Cell::new(0) };
}

/// Initialise the per-thread RNG from the global counter + an address for
/// ASLR entropy. Called lazily on first `make_tool_call_id()`.
fn ensure_rng_seeded() {
    RNG_STATE.with(|cell| {
        if cell.get() == 0 {
            let global = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
            let stack_ptr = std::ptr::addr_of!(global) as u64 & 0xFFFF_FFFF;
            // Mix the global counter with the stack address for entropy.
            let seed = global
                .wrapping_mul(0x517c_c1b7_2722_0a95)
                .wrapping_add(stack_ptr);
            cell.set(seed | 1); // Ensure non-zero.
        }
    });
}

/// Xorshift64 step — fast, no allocation, good enough for IDs.
fn xorshift64(state: u64) -> u64 {
    let mut x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Generate a random-looking tool-call id. vLLM uses `make_tool_call_id()`
/// which produces a 9-char alphanumeric string prefixed with `call_`. We
/// match that format so OpenAI clients can key tool results on the id.
///
/// (B7) Replaces the old deterministic `call_aaaaaaaa` with a per-call
/// random value derived from a thread-local xorshift PRNG seeded from a
/// global counter and ASLR address entropy.
fn make_tool_call_id() -> String {
    ensure_rng_seeded();

    let val = RNG_STATE.with(|cell| {
        let s = xorshift64(cell.get());
        cell.set(s);
        s
    });

    // Encode as 8-char base36 (a-z0-9) to stay within alphanumeric charset.
    let charset = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut s = String::with_capacity(13);
    s.push_str("call_");
    let mut v = val;
    for _ in 0..8 {
        s.push(char::from(charset[(v % 36) as usize]));
        v /= 36;
    }
    s
}

/// What kind of parser a dialect is — tool-call or reasoning.
/// Used by the registry and shims to pick the correct engine interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserKind {
    /// Tool-call parser (registered via tool-parser interfaces).
    Tool,
    /// Reasoning parser (registered via reasoning-parser interfaces).
    Reasoning,
}

impl ParserKind {
    /// Return the kind as a static string: "tool" or "reasoning".
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::Reasoning => "reasoning",
        }
    }
}

/// A parse error — something went wrong that the engine should surface.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParseError {
    /// The dialect name was not found in the registry.
    #[error("unknown dialect: {0}")]
    UnknownDialect(String),

    /// The token text could not be parsed by the dialect's state machine.
    #[error("parse error in dialect {dialect}: {detail}")]
    Malformed {
        /// Which dialect produced the error.
        dialect: &'static str,
        /// Human-readable detail.
        detail: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_ids_are_unique() {
        let mut ids = Vec::new();
        for _ in 0..100 {
            let id = make_tool_call_id();
            assert!(id.starts_with("call_"));
            assert_eq!(id.len(), 13, "id should be call_ + 8 chars");
            ids.push(id);
        }
        // All 100 should be unique (extremely likely with 36^8 space).
        let unique: std::collections::HashSet<_> = ids.into_iter().collect();
        assert_eq!(unique.len(), 100, "IDs should be unique");
    }

    #[test]
    fn tool_call_ids_are_alphanumeric() {
        for _ in 0..50 {
            let id = make_tool_call_id();
            assert!(id.starts_with("call_"));
            for c in id["call_".len()..].chars() {
                assert!(
                    c.is_ascii_alphanumeric(),
                    "id char '{c}' should be alphanumeric"
                );
            }
        }
    }

    #[test]
    fn begin_tool_call_generates_different_ids() {
        let mut state = ToolCallArgState::new();
        state.begin_tool_call();
        let id1 = state.id.clone().unwrap();
        state.complete_tool_call();
        state.reset_to_idle();
        state.begin_tool_call();
        let id2 = state.id.clone().unwrap();
        assert_ne!(id1, id2, "consecutive tool calls should have different IDs");
    }
}
