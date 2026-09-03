//! Built-in dialect implementations.
//!
//! Each module implements one model dialect's tool-call / reasoning format.
//! The implementations are streaming state machines — they recognise markers
//! in the text and emit deltas as content crosses the marker boundary.

pub mod deepseek;
pub mod hermes;
pub mod llama3_json;

pub use deepseek::DeepSeekReasoningParser;
pub use hermes::HermesParser;
pub use llama3_json::Llama3JsonParser;
