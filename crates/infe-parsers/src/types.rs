//! Delta types emitted by the streaming parser.
//!
//! These mirror the `OpenAI` streaming chunk format so they can be serialised
//! directly into the SSE stream by the API server with zero translation.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]

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
        self.id = Some(make_tool_call_id(self.index.unwrap_or(0)));
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

/// Generate a deterministic tool-call id. vLLM uses `make_tool_call_id()`
/// which produces a 9-char alphanumeric string prefixed with `call_`. We
/// match that format so OpenAI clients can key tool results on the id.
fn make_tool_call_id(index: usize) -> String {
    // Simple deterministic id from the index — good enough for streaming.
    // Real engines generate random ids; the format is what matters for parity.
    let seed = index.wrapping_mul(2_654_435_761);
    let mut s = String::with_capacity(13);
    s.push_str("call_");
    for i in 0..8 {
        let n = ((seed >> (i * 4)) & 0x1f) as u32;
        let c = if n < 26 {
            char::from_u32(u32::from(b'a') + n).unwrap_or('a')
        } else {
            char::from_u32(u32::from(b'0') + (n - 26)).unwrap_or('0')
        };
        s.push(c);
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
