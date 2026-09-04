//! Llama-3 JSON tool-call parser.
//!
//! Llama-3.1+ models use a JSON object with "name" and "parameters" fields,
//! terminated by the model's end-of-turn token. This parser mirrors vLLM's
//! `llama3_json_tool_parser.py`:
//!
//! 1. Detect the opening `{` that starts the JSON object.
//! 2. Accumulate the JSON body incrementally.
//! 3. Extract the `name` via regex as soon as it's available — emit name+id.
//! 4. Extract the `parameters` value (everything after `"parameters":`) and
//!    compute the diff against what was already sent — emit the diff.
//! 5. At completion (matching `}`), emit any remaining diff + completion.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]

use crate::parser::{DialectParser, ParseResult};
use crate::types::{ToolCallArgState, ToolCallDelta, ToolCallState};

#[derive(Debug, Default)]
pub struct Llama3JsonParser {
    brace_depth: u32,
    accumulating: bool,
}

impl Llama3JsonParser {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract the tool name from a (possibly partial) JSON body.
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

    /// Extract the raw parameters text — everything after `"parameters":`
    /// (or `"arguments":` as a fallback). When `is_complete`, strips the
    /// trailing `}` that closes the wrapper.
    fn extract_params_raw(json_str: &str, is_complete: bool) -> Option<String> {
        let key = if json_str.contains("\"parameters\"") {
            "\"parameters\""
        } else {
            "\"arguments\""
        };
        let idx = json_str.find(key)?;
        let after_key = &json_str[idx + key.len()..];
        let colon_idx = after_key.find(':')?;
        let raw = &after_key[colon_idx + 1..];
        let mut raw = raw.trim_start().to_string();

        if is_complete {
            raw = raw.trim_end().to_string();
            if raw.ends_with('}') {
                raw.truncate(raw.len() - 1);
                raw = raw.trim_end().to_string();
            }
        }
        Some(raw)
    }

    fn is_complete_json(json_str: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(json_str).is_ok()
    }

    /// Try to emit incremental name + arg diffs.
    fn try_emit_incremental(
        state: &mut ToolCallArgState,
        json_body: &str,
        result: &mut ParseResult,
    ) {
        let is_complete = Self::is_complete_json(json_body);

        if !state.name_emitted {
            if let Some(name) = Self::extract_name(json_body) {
                state.name_emitted = true;
                result.tool_calls.push(ToolCallDelta {
                    index: state.index.unwrap_or(0),
                    id: state.id.take(),
                    name: Some(name),
                    arguments_fragment: String::new(),
                    is_complete: false,
                });
            } else {
                return;
            }
        }

        if let Some(params_raw) = Self::extract_params_raw(json_body, is_complete) {
            let already = state.last_emitted_args_len;
            if params_raw.len() > already {
                let diff = &params_raw[already..];
                state.last_emitted_args_len = params_raw.len();
                result.tool_calls.push(ToolCallDelta {
                    index: state.index.unwrap_or(0),
                    id: None,
                    name: None,
                    arguments_fragment: diff.to_string(),
                    is_complete: false,
                });
            }
        }
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

                        // Try immediate incremental emission
                        let buf = state.arguments_buffer.clone();
                        Self::try_emit_incremental(state, &buf, result);
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
                                let json_body = state.arguments_buffer.clone();

                                // Emit final incremental diffs.
                                Self::try_emit_incremental(state, &json_body, result);

                                // Emit completion delta with any remaining args.
                                if let Some(params_raw) = Self::extract_params_raw(&json_body, true)
                                {
                                    let already = state.last_emitted_args_len;
                                    if params_raw.len() > already {
                                        let diff = &params_raw[already..];
                                        result.tool_calls.push(ToolCallDelta {
                                            index: state.index.unwrap_or(0),
                                            id: None,
                                            name: None,
                                            arguments_fragment: diff.to_string(),
                                            is_complete: true,
                                        });
                                        state.last_emitted_args_len = params_raw.len();
                                    } else {
                                        result.tool_calls.push(ToolCallDelta {
                                            index: state.index.unwrap_or(0),
                                            id: None,
                                            name: None,
                                            arguments_fragment: String::new(),
                                            is_complete: true,
                                        });
                                    }
                                } else {
                                    result.tool_calls.push(ToolCallDelta {
                                        index: state.index.unwrap_or(0),
                                        id: None,
                                        name: None,
                                        arguments_fragment: String::new(),
                                        is_complete: true,
                                    });
                                }

                                state.complete_tool_call();
                                state.reset_to_idle();
                                self.accumulating = false;
                            } else {
                                // Nested — try incremental emission.
                                let buf = state.arguments_buffer.clone();
                                Self::try_emit_incremental(state, &buf, result);
                            }
                        }
                        _ => {
                            let buf = state.arguments_buffer.clone();
                            Self::try_emit_incremental(state, &buf, result);
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
    use crate::types::ToolCallDelta;
    fn make_parser() -> StreamingParser {
        let dialect = DialectRegistry::create("llama3_json").unwrap();
        StreamingParser::new(dialect)
    }

    fn accumulate_args(deltas: &[ToolCallDelta]) -> String {
        deltas
            .iter()
            .map(|d| d.arguments_fragment.as_str())
            .collect()
    }

    fn find_name(deltas: &[ToolCallDelta]) -> Option<&str> {
        deltas.iter().find_map(|d| d.name.as_deref())
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
        assert_eq!(find_name(&r.tool_calls).unwrap(), "get_weather");
        let args = accumulate_args(&r.tool_calls);
        assert!(args.contains("\"London\""));
        assert!(!args.contains("\"name\""));
    }

    #[test]
    fn json_tool_call_generates_id() {
        let mut p = make_parser();
        let r = p.feed(r#"{"name":"fn","parameters":{}}"#);
        assert!(!r.tool_calls.is_empty());
        assert!(r.tool_calls[0].id.is_some());
        assert!(r.tool_calls[0].id.as_ref().unwrap().starts_with("call_"));
    }

    #[test]
    fn multiple_json_calls_get_incrementing_indexes() {
        let mut p = make_parser();
        let _ = p.feed(r#"{"name":"fn1","parameters":{"a":1}}"#);
        let r = p.feed(r#"{"name":"fn2","parameters":{"b":2}}"#);
        let completed: Vec<_> = r.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].index, 1);
        assert_eq!(find_name(&r.tool_calls).unwrap(), "fn2");
    }

    #[test]
    fn json_tool_call_streamed() {
        let mut p = make_parser();
        let _r1 = p.feed(r#"{"name":"#);
        let r_name = p.feed(r#""get_weather","#);
        // Name should be emitted when the closing quote of the name value arrives
        assert_eq!(find_name(&r_name.tool_calls).unwrap(), "get_weather");
        let _r3 = p.feed(r#""parameters":{"city":"#);
        let r4 = p.feed(r#""London"}}"#);
        assert!(!r4.tool_calls.is_empty());
    }

    #[test]
    fn nested_json_braces() {
        let mut p = make_parser();
        let r = p.feed(r#"{"name":"fn","parameters":{"nested":{"a":1}}}"#);
        let completed: Vec<_> = r.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert_eq!(completed.len(), 1);
        assert!(completed[0].is_complete);
    }

    #[test]
    fn content_before_and_after_json() {
        let mut p = make_parser();
        let r1 = p.feed("Sure! ");
        assert!(!r1.content.is_empty());
        let r2 = p.feed(r#"{"name":"fn","parameters":{}}"#);
        let completed: Vec<_> = r2.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert_eq!(completed.len(), 1);
        assert!(completed[0].is_complete);
        let r3 = p.feed(" Done.");
        assert!(!r3.content.is_empty());
    }

    #[test]
    fn incremental_argument_streaming() {
        let mut p = make_parser();
        let _ = p.feed(r#"{"name":"fn","parameters":"#);
        let mut all_args = String::new();
        for piece in [r#"{"city":"#, r#""London"}}"#] {
            let r = p.feed(piece);
            for tc in &r.tool_calls {
                all_args.push_str(&tc.arguments_fragment);
            }
        }
        assert!(
            all_args.contains("London"),
            "Accumulated args should contain London, got: {all_args}"
        );
        assert!(
            all_args.contains("\"city\""),
            "Accumulated args should contain city key, got: {all_args}"
        );
    }
}
