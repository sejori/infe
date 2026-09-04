//! Llama-3 JSON tool-call parser.
//!
//! Llama-3.1+ models use the built-in tool-call format where the model emits
//! a JSON object with "name" and "parameters" fields, terminated by the
//! model's end-of-turn token. This parser detects the opening brace of the
//! JSON object, streams the parameters as arguments fragments, and emits
//! a complete delta when the closing brace is seen.
//!
//! The `arguments` delta is the value of the "parameters" field (not the
//! whole JSON object), matching vLLM's `llama3_json_tool_parser.py`.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]

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

    /// Extract the `name` field and the `parameters` sub-object from the
    /// Llama-3 JSON body `{"name":"fn","parameters":{...}}`.
    ///
    /// Returns `(name, parameters_json_str)` where `parameters_json_str` is
    /// the serialised value of the "parameters" field.
    fn extract_name_and_params(json_str: &str) -> (Option<String>, String) {
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(json_str) {
            let name = obj
                .get("name")
                .and_then(|v| v.as_str())
                .map(ToString::to_string);
            let params = match obj.get("parameters").or_else(|| obj.get("arguments")) {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => "{}".to_string(),
            };
            return (name, params);
        }
        // Fallback: manual extraction for partial JSON.
        if let Some(name_idx) = json_str.find("\"name\"") {
            let after_key = &json_str[name_idx + 6..];
            if let Some(colon_idx) = after_key.find(':') {
                let after_colon = after_key[colon_idx + 1..].trim_start();
                if let Some(rest) = after_colon.strip_prefix('"') {
                    if let Some(end) = rest.find('"') {
                        return (Some(rest[..end].to_string()), json_str.to_string());
                    }
                }
            }
        }
        (None, json_str.to_string())
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
                        state.begin_tool_call();
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
                                // Complete tool call.
                                let (name, params) =
                                    Self::extract_name_and_params(&state.arguments_buffer);

                                // Compute final arguments diff.
                                let already = state.last_emitted_args_len.min(params.len());
                                let args_diff = &params[already..];

                                let first_delta = !state.name_emitted;
                                result.tool_calls.push(ToolCallDelta {
                                    index: state.index.unwrap_or(0),
                                    id: state.id.take(),
                                    name: if first_delta { name } else { None },
                                    arguments_fragment: args_diff.to_string(),
                                    is_complete: true,
                                });
                                state.name_emitted = true;
                                state.last_emitted_args_len = params.len();

                                state.complete_tool_call();
                                state.reset_to_idle();
                                self.accumulating = false;
                            } else {
                                // Still accumulating inside nested JSON.
                                // Don't emit intermediate deltas — the
                                // completion delta sends the extracted value.
                            }
                        }
                        _ => {
                            // Accumulating — opportunistically emit diffs.
                            // We accumulate the raw buffer and compute diffs.
                            // The completion delta will send the correct
                            // extracted value.
                            // Intentionally don't emit partial deltas for
                            // Llama3 JSON — the body is single-chunk usually.
                        }
                    }
                }
                ToolCallState::Complete => {
                    state.reset_to_idle();
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
        let completed: Vec<_> = r.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert_eq!(completed.len(), 1);
        assert!(completed[0].is_complete);
        assert_eq!(completed[0].name.as_deref(), Some("get_weather"));
        // Should extract parameters, not the whole wrapper.
        let args = &completed[0].arguments_fragment;
        assert!(args.contains("\"London\""));
        assert!(!args.contains("\"name\""));
    }

    #[test]
    fn json_tool_call_generates_id() {
        let mut p = make_parser();
        let r = p.feed(r#"{"name":"fn","parameters":{}}"#);
        assert!(!r.tool_calls.is_empty());
        let first = &r.tool_calls[0];
        assert!(first.id.is_some());
        assert!(first.id.as_ref().unwrap().starts_with("call_"));
    }

    #[test]
    fn multiple_json_calls_get_incrementing_indexes() {
        let mut p = make_parser();
        let _ = p.feed(r#"{"name":"fn1","parameters":{"a":1}}"#);
        let r = p.feed(r#"{"name":"fn2","parameters":{"b":2}}"#);
        let completed: Vec<_> = r.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].index, 1);
        assert_eq!(completed[0].name.as_deref(), Some("fn2"));
    }

    #[test]
    fn json_tool_call_streamed() {
        let mut p = make_parser();
        let _r1 = p.feed(r#"{"name":"#);
        let _r2 = p.feed(r#""get_weather","#);
        let _r3 = p.feed(r#""parameters":{"city":"#);
        let r4 = p.feed(r#""London"}}"#);
        assert!(!r4.tool_calls.is_empty());
        let completed = r4.tool_calls.iter().find(|d| d.is_complete).unwrap();
        assert_eq!(completed.name.as_deref(), Some("get_weather"));
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
