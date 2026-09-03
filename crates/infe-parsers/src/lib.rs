//! # infe-parsers
//!
//! Streaming tool-call and reasoning-content parsers for LLM inference.
//!
//! The streaming parser is called once per engine step with a batch of
//! newly-decoded token-text chunks. It produces structured deltas —
//! [`ToolCallDelta`]s and [`ReasoningDelta`]s — in the `OpenAI` streaming
//! format, without per-token Python calls.
//!
//! Each model dialect (Hermes, Llama-3 JSON, Qwen, `DeepSeek`, …) is a
//! data-driven state machine implementing [`DialectParser`]. Dialects are
//! registered at build time and selected by name at engine start.
//!
//! See `BRIEF.md` §6.1 for the design rationale.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod dialects;
pub mod parser;
pub mod registry;
pub mod types;

pub use parser::{DialectParser, ParseResult, StreamingParser};
pub use registry::DialectRegistry;
pub use types::{ParseError, ReasoningDelta, ToolCallArgState, ToolCallDelta, ToolCallState};

/// The component name, as used in the manifest and `StepComponent` trait.
pub const COMPONENT_NAME: &str = "infe-parsers";
