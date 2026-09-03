//! The streaming parser trait and the per-request parsing session.
//!
//! The [`StreamingParser`] is the top-level entry point. The engine creates
//! one instance per request at request start, selects a dialect by name, and
//! calls [`StreamingParser::feed`] with each batch of decoded text chunks.
//! The parser returns a [`ParseResult`] containing zero or more deltas.
//!
//! Unlike the per-token Python parsers in vLLM/SGLang, this processes a batch
//! of text chunks in a single call, reducing Python↔Rust crossings.

use crate::types::ToolCallArgState;

/// The result of feeding text into the parser: zero or more deltas.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParseResult {
    /// Tool-call deltas, in stream order.
    pub tool_calls: Vec<crate::types::ToolCallDelta>,
    /// Reasoning deltas, in stream order.
    pub reasoning: Vec<crate::types::ReasoningDelta>,
    /// Text fragments that are neither tool-call nor reasoning — plain content
    /// that should be passed through as `delta.content`.
    pub content: Vec<String>,
}

impl ParseResult {
    /// Create an empty result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether this result has any deltas at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tool_calls.is_empty() && self.reasoning.is_empty() && self.content.is_empty()
    }
}

/// The dialect parser trait — one implementation per model dialect.
///
/// Each dialect is a state machine that recognises tool-call and reasoning
/// markers in token text and emits structured deltas.
pub trait DialectParser: Send + Sync {
    /// The dialect name (e.g. `"hermes"`, `"llama3_json"`).
    fn name(&self) -> &'static str;

    /// Feed a chunk of decoded text and return any deltas produced.
    ///
    /// The text is a fragment of the model's decoded output — not a full
    /// token, because the engine may coalesce tokens before handing them
    /// to the parser. The parser must handle partial markers that span
    /// chunk boundaries by buffering internally.
    fn feed(&mut self, text: &str, state: &mut ToolCallArgState, result: &mut ParseResult);

    /// Reset the parser to its initial state (e.g. for a new request).
    fn reset(&mut self, state: &mut ToolCallArgState);
}

/// The top-level streaming parser, holding per-request state.
///
/// The engine creates one of these per request, selects a dialect, and calls
/// `feed` with each batch of decoded text. This is the type that crosses the
/// `PyO3` boundary.
pub struct StreamingParser {
    /// The dialect-specific parser.
    dialect: Box<dyn DialectParser>,
    /// Per-request tool-call state.
    state: ToolCallArgState,
}

impl StreamingParser {
    /// Create a new parser with the given dialect.
    #[must_use]
    pub fn new(dialect: Box<dyn DialectParser>) -> Self {
        Self {
            dialect,
            state: ToolCallArgState::default(),
        }
    }

    /// Feed a chunk of decoded text and return the deltas produced.
    ///
    /// This is the hot path — called once per engine step (or batch of
    /// decoded tokens), not once per token.
    pub fn feed(&mut self, text: &str) -> ParseResult {
        let mut result = ParseResult::new();
        self.dialect.feed(text, &mut self.state, &mut result);
        result
    }

    /// Feed a batch of text chunks (multiple decoded tokens coalesced).
    ///
    /// This is the preferred entry point for step-granular calls: the engine
    /// passes all tokens decoded this step in one call, avoiding per-token
    /// crossings.
    pub fn feed_batch(&mut self, chunks: &[&str]) -> ParseResult {
        let mut combined = ParseResult::new();
        for chunk in chunks {
            let r = self.feed(chunk);
            combined.tool_calls.extend(r.tool_calls);
            combined.reasoning.extend(r.reasoning);
            combined.content.extend(r.content);
        }
        combined
    }

    /// Reset the parser for a new request.
    pub fn reset(&mut self) {
        self.dialect.reset(&mut self.state);
    }
}
