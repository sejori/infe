//! Delta types emitted by the streaming parser.
//!
//! These mirror the `OpenAI` streaming chunk format so they can be serialised
//! directly into the SSE stream by the API server with zero translation.

#![allow(clippy::module_name_repetitions)]

use serde::{Deserialize, Serialize};

/// A structured tool-call delta, emitted as the parser discovers tool-call
/// content in the token stream.
///
/// This matches the `delta.tool_calls[].function.arguments` field in the
/// `OpenAI` streaming protocol. Each delta is a *fragment* — the API server
/// appends it to the accumulated arguments string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Index of the tool call in the batch (0-based, matching `OpenAI`'s
    /// streaming `index` field).
    pub index: usize,

    /// Optional tool-call id (only present on the first delta for this call).
    pub id: Option<String>,

    /// Optional function name (only present on the first delta, extracted
    /// from the dialect's name field — e.g. Hermes `<tool_call>` JSON).
    pub name: Option<String>,

    /// Fragment of the arguments JSON string. May be a partial JSON fragment
    /// (the consumer accumulates and the API server can stream it raw).
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
    /// Inside the tool-call block, before the JSON arguments field.
    InToolCall,
    /// Inside the arguments string, accumulating fragments.
    InArguments,
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

/// State of a single tool-call argument parser (per-request, per-tool-call).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToolCallArgState {
    /// Current parser state.
    pub state: ToolCallState,

    /// Accumulated function name (until the arguments field begins).
    pub name_buffer: String,

    /// Accumulated arguments JSON string (for internal bookkeeping / testing).
    pub arguments_buffer: String,

    /// Tool-call id, if assigned.
    pub id: Option<String>,

    /// Index assigned when the first delta is emitted.
    pub index: Option<usize>,
}

impl ToolCallArgState {
    /// Create a new idle state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
