//! Llama-3 JSON tool-call parser.
//!
//! Llama-3.1+ models use the built-in tool-call format where the model emits
//! a JSON object with "name" and "parameters" fields, terminated by the
//! model's end-of-turn token. This parser detects the opening brace of the
//! JSON object, streams the parameters as arguments fragments, and emits
//! a complete delta when the closing brace is seen.

#![allow(clippy::cast_possible_truncation)]

use crate::parser::{DialectParser, ParseResult};
use crate::types::{ToolCallArgState, ToolCallDelta, ToolCallState};

/// Depth of braces seen so far (for nested JSON).
#[derive(Debug, Default)]
pub struct Llama3JsonParser {
    /// Brace depth: 0 = outside JSON, >0 = inside.
    brace_depth: u32,
    /// Whether we have seen the opening brace and are accumulating.
    accumulating: bool,
}

impl Llama3JsonParser {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn extract_name(json_str: &str) -> Option<String> {
        if let Some(name_idx) = json_str.find("\"name\"") {
            let after_key = &json_str[name_idx + 6..];
            if let Some(colon_idx) = after_key.find(':') {
                let after_colon = after_key[colon_idx + 1..].trim_start();
                if let Some(rest) = after_colon.strip_prefix('"') {
                    if let Some(end) = rest.find('"') {
                        return Some(rest[..end].to_string());
                    }
                }
            }
        }
        None
    }
}

impl DialectParser for Llama3JsonParser {
    fn name(&self) -> &'static str {
        "llama3_json"
    }

    fn feed(&mut self, text: &str, state: &mut ToolCallArgState, result: &mut ParseResult) {
        for ch in text.chars() {
            match state.state {
                ToolCallState::Idle => {
                    if ch == '{' && !self.accumulating {
                        self.accumulating = true;
                        self.brace_depth = 1;
                        state.state = ToolCallState::InToolCall;
                        state.arguments_buffer.push(ch);
                    } else {
                        result.content.push(ch.to_string());
                    }
                }
                ToolCallState::InToolCall => {
                    state.arguments_buffer.push(ch);
                    match ch {
                        '{' => {
                            self.brace_depth += 1;
                        }
                        '}' => {
                            self.brace_depth -= 1;
                            if self.brace_depth == 0 {
                                let name = Self::extract_name(&state.arguments_buffer);
                                let delta = ToolCallDelta {
                                    index: state.index.unwrap_or(0),
                                    id: state.id.take(),
                                    name,
                                    arguments_fragment: state.arguments_buffer.clone(),
                                    is_complete: true,
                                };
                                result.tool_calls.push(delta);
                                state.state = ToolCallState::Complete;
                                self.accumulating = false;
                            }
                        }
                        _ => {}
                    }
                }
                ToolCallState::Complete => {
                    result.content.push(ch.to_string());
                }
                ToolCallState::InArguments => {
                    state.state = ToolCallState::InToolCall;
                }
            }
        }
    }

    fn reset(&mut self, state: &mut ToolCallArgState) {
        *state = ToolCallArgState::new();
        self.brace_depth = 0;
        self.accumulating = false;
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::StreamingParser;
    use crate::registry::DialectRegistry;
    fn make_parser() -> StreamingParser {
        let dialect = DialectRegistry::create("llama3_json").unwrap();
        StreamingParser::new(dialect)
    }

    #[test]
    fn plain_content_passes_through() {
        let mut p = make_parser();
        let r = p.feed("Hello, world!");
        assert!(r.tool_calls.is_empty());
        assert!(!r.content.is_empty());
        let joined: String = r.content.join("");
        assert!(joined.contains("Hello"));
    }

    #[test]
    fn json_tool_call_single_chunk() {
        let mut p = make_parser();
        let r = p.feed(r#"{"name":"get_weather","parameters":{"city":"London"}}"#);
        assert_eq!(r.tool_calls.len(), 1);
        assert!(r.tool_calls[0].is_complete);
        assert_eq!(r.tool_calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn json_tool_call_streamed() {
        let mut p = make_parser();
        let _r1 = p.feed(r#"{"name":"#);
        let _r2 = p.feed(r#""get_weather","#);
        let _r3 = p.feed(r#""parameters":{"city":"#);
        let r4 = p.feed(r#""London"}}"#);
        assert!(!r4.tool_calls.is_empty());
        assert!(r4.tool_calls[0].is_complete);
        assert_eq!(r4.tool_calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn nested_json_braces() {
        let mut p = make_parser();
        let r = p.feed(r#"{"name":"fn","parameters":{"nested":{"a":1}}}"#);
        assert_eq!(r.tool_calls.len(), 1);
        assert!(r.tool_calls[0].is_complete);
    }

    #[test]
    fn content_before_and_after_json() {
        let mut p = make_parser();
        let r1 = p.feed("Sure! ");
        assert!(!r1.content.is_empty());
        let r2 = p.feed(r#"{"name":"fn","parameters":{}}"#);
        assert_eq!(r2.tool_calls.len(), 1);
        assert!(r2.tool_calls[0].is_complete);
        let r3 = p.feed(" Done.");
        assert!(!r3.content.is_empty());
    }
}
