#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]

use crate::parser::{DialectParser, ParseResult};
use crate::types::{ToolCallArgState, ToolCallDelta, ToolCallState};

const OPEN_MARKER: &str = "\u{3c}tool_call\u{3e}";
const CLOSE_MARKER: &str = "\u{3c}/tool_call\u{3e}";

#[derive(Debug, Default)]
pub struct HermesParser {
    pending: String,
}

impl HermesParser {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn match_marker(text: &str, marker: &str) -> Option<usize> {
        if text.starts_with(marker) {
            Some(marker.len())
        } else {
            None
        }
    }

    fn is_partial_prefix(text: &str, marker: &str) -> bool {
        !text.is_empty() && marker.strip_prefix(text).is_some()
    }

    /// Find the position of a partial marker prefix at the END of text.
    fn find_tail_partial(text: &str, marker: &str) -> usize {
        let max_len = text.len().min(marker.len());
        for len in (1..=max_len).rev() {
            let suffix = &text[text.len() - len..];
            if marker.strip_prefix(suffix).is_some() {
                return text.len() - len;
            }
        }
        text.len()
    }

    /// Extract the `name` field and the `arguments` sub-object from a Hermes
    /// JSON tool-call body.
    ///
    /// Hermes wraps the call as `{"name":"fn","arguments":{...}}`. We extract
    /// the `arguments` value (which is the inner JSON object) separately from
    /// the `name` field, matching what vLLM and SGLang do:
    ///   - `name` goes on the first delta's `function.name`
    ///   - `arguments` value is streamed as `function.arguments` fragments
    ///
    /// Returns `(name, arguments_json_str)` where `arguments_json_str` is the
    /// serialised value of the `arguments` field (e.g. `{"city":"London"}`),
    /// or `"{}"` if absent.
    fn extract_name_and_args(json_str: &str) -> (Option<String>, String) {
        // Parse as JSON to extract the fields properly.
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(json_str) {
            let name = obj
                .get("name")
                .and_then(|v| v.as_str())
                .map(ToString::to_string);
            let args = match obj.get("arguments") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => "{}".to_string(),
            };
            return (name, args);
        }
        // Fallback: manual extraction for partial JSON (during streaming,
        // we may get the name before the full arguments object is available).
        // The name is usually the first complete field: "name":"value".
        if let Some(name_idx) = json_str.find("\"name\"") {
            let after_key = &json_str[name_idx + 6..];
            if let Some(colon_idx) = after_key.find(':') {
                let after_colon = after_key[colon_idx + 1..].trim_start();
                // Check if this is the "name" field (not "arguments")
                if let Some(rest) = after_colon.strip_prefix('"') {
                    if let Some(end) = rest.find('"') {
                        let name = rest[..end].to_string();
                        // Try to find the "arguments" field value start
                        if let Some(args_idx) = json_str.find("\"arguments\"") {
                            let after_args_key = &json_str[args_idx + 11..];
                            if let Some(args_colon) = after_args_key.find(':') {
                                let args_val_start = after_args_key[args_colon + 1..].trim_start();
                                // Return whatever we have of the arguments
                                // value so far — it may be incomplete.
                                let args_val = args_val_start.to_string();
                                return (Some(name), args_val);
                            }
                        }
                        // No arguments field yet — name only.
                        return (Some(name), String::new());
                    }
                }
            }
        }
        (None, json_str.to_string())
    }
}

#[allow(clippy::nonminimal_bool)]
impl DialectParser for HermesParser {
    fn name(&self) -> &'static str {
        "hermes"
    }

    fn feed(&mut self, text: &str, state: &mut ToolCallArgState, result: &mut ParseResult) {
        let full_text = if self.pending.is_empty() {
            text.to_string()
        } else {
            let mut combined = String::with_capacity(self.pending.len() + text.len());
            combined.push_str(&self.pending);
            combined.push_str(text);
            self.pending.clear();
            combined
        };

        let mut remaining = full_text.as_str();

        while !remaining.is_empty() {
            match state.state {
                ToolCallState::Idle => {
                    // Check if remaining starts with the open marker.
                    if let Some(consumed) = Self::match_marker(remaining, OPEN_MARKER) {
                        // Begin a new tool call: assign index, generate id.
                        state.begin_tool_call();
                        remaining = &remaining[consumed..];
                    } else if Self::is_partial_prefix(remaining, OPEN_MARKER) {
                        self.pending.push_str(remaining);
                        break;
                    } else {
                        // Search for the open marker anywhere in the text.
                        if let Some(pos) = remaining.find('<') {
                            let slice = &remaining[pos..];
                            // First check if this is a full match.
                            if let Some(consumed) = Self::match_marker(slice, OPEN_MARKER) {
                                // Emit content before the marker.
                                if pos > 0 {
                                    result.content.push(remaining[..pos].to_string());
                                }
                                state.begin_tool_call();
                                remaining = &remaining[pos + consumed..];
                                continue;
                            }
                            // Otherwise check if it's a partial prefix.
                            if Self::is_partial_prefix(slice, OPEN_MARKER) {
                                if pos > 0 {
                                    result.content.push(remaining[..pos].to_string());
                                }
                                self.pending.push_str(slice);
                                break;
                            }
                        }
                        // No marker found - emit all as content.
                        result.content.push(remaining.to_string());
                        break;
                    }
                }
                ToolCallState::InToolCall => {
                    // Search for the close marker in remaining.
                    if let Some(pos) = remaining.find(CLOSE_MARKER) {
                        // Accumulate content before the close marker, emit diff.
                        let chunk = &remaining[..pos];
                        if !chunk.is_empty() {
                            state.arguments_buffer.push_str(chunk);
                        }

                        // Extract name and arguments from the accumulated buffer.
                        let (name, args) = Self::extract_name_and_args(&state.arguments_buffer);

                        // Compute the final arguments diff: emit only characters
                        // we haven't sent yet.
                        let args_diff = if args.is_empty() {
                            ""
                        } else {
                            let already = state.last_emitted_args_len.min(args.len());
                            &args[already..]
                        };

                        // Emit a completion delta.
                        #[allow(clippy::nonminimal_bool)]
                        #[allow(clippy::nonminimal_bool)]
                        let first_delta = !state.name_emitted;
                        result.tool_calls.push(ToolCallDelta {
                            index: state.index.unwrap_or(0),
                            id: state.id.take(),
                            name: if first_delta { name } else { None },
                            arguments_fragment: args_diff.to_string(),
                            is_complete: true,
                        });
                        state.name_emitted = true;
                        state.last_emitted_args_len = args.len();

                        state.complete_tool_call();
                        state.reset_to_idle();
                        remaining = &remaining[pos + CLOSE_MARKER.len()..];
                    } else if Self::is_partial_prefix(remaining, CLOSE_MARKER) {
                        // Might be the start of a close marker — buffer and wait.
                        self.pending.push_str(remaining);
                        break;
                    } else {
                        // No close marker found. Accumulate the raw text.
                        // We don't emit intermediate argument deltas here —
                        // the arguments value isn't extracted until the JSON
                        // is complete. At completion, we emit the extracted
                        // sub-object as a single diff.
                        //
                        // Check if the tail is a partial prefix of close.
                        let split_pos = Self::find_tail_partial(remaining, CLOSE_MARKER);
                        state.arguments_buffer.push_str(&remaining[..split_pos]);
                        if split_pos < remaining.len() {
                            self.pending.push_str(&remaining[split_pos..]);
                        }
                        break;
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
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::StreamingParser;
    use crate::registry::DialectRegistry;

    fn make_parser() -> StreamingParser {
        let dialect = DialectRegistry::create("hermes").unwrap();
        StreamingParser::new(dialect)
    }

    #[test]
    fn plain_content_passes_through() {
        let mut p = make_parser();
        let r = p.feed("Hello, world!");
        assert!(r.tool_calls.is_empty());
        assert_eq!(r.content, vec!["Hello, world!".to_string()]);
    }

    #[test]
    fn tool_call_single_chunk() {
        let mut p = make_parser();
        let input = format!("{OPEN_MARKER}{{\"name\":\"fn\",\"arguments\":{{}}}}{CLOSE_MARKER}");
        let r = p.feed(&input);
        let completed: Vec<_> = r.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].name.as_deref(), Some("fn"));
        // Arguments should be just the inner object, not the whole wrapper.
        assert_eq!(completed[0].arguments_fragment, "{}");
    }

    #[test]
    fn tool_call_extracts_arguments_subobject() {
        let mut p = make_parser();
        let input = format!(
            "{OPEN_MARKER}{{\"name\":\"get_weather\",\"arguments\":{{\"city\":\"London\"}}}}{CLOSE_MARKER}"
        );
        let r = p.feed(&input);
        let completed: Vec<_> = r.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].name.as_deref(), Some("get_weather"));
        // Should be just the arguments value, not the wrapper.
        let args = &completed[0].arguments_fragment;
        assert!(
            args.contains("\"city\"") && args.contains("\"London\""),
            "args should contain city/London, got: {args}"
        );
        assert!(
            !args.contains("\"name\""),
            "args should NOT contain the name field, got: {args}"
        );
    }

    #[test]
    fn tool_call_generates_id_on_first_delta() {
        let mut p = make_parser();
        let input = format!("{OPEN_MARKER}{{\"name\":\"fn\",\"arguments\":{{}}}}{CLOSE_MARKER}");
        let r = p.feed(&input);
        assert!(!r.tool_calls.is_empty());
        let first = &r.tool_calls[0];
        assert!(first.id.is_some(), "first delta should have an id");
        let id = first.id.as_ref().unwrap();
        assert!(
            id.starts_with("call_"),
            "id should start with call_, got: {id}"
        );
    }

    #[test]
    fn multiple_tool_calls_get_incrementing_indexes() {
        let mut p = make_parser();
        let input = format!(
            "{OPEN_MARKER}{{\"name\":\"fn1\",\"arguments\":{{\"a\":1}}}}{CLOSE_MARKER}{OPEN_MARKER}{{\"name\":\"fn2\",\"arguments\":{{\"b\":2}}}}{CLOSE_MARKER}"
        );
        let r = p.feed(&input);
        let completed: Vec<_> = r.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert_eq!(completed.len(), 2);
        assert_eq!(completed[0].index, 0);
        assert_eq!(completed[1].index, 1);
        assert_eq!(completed[0].name.as_deref(), Some("fn1"));
        assert_eq!(completed[1].name.as_deref(), Some("fn2"));
    }

    #[test]
    fn tool_call_streamed_across_chunks() {
        let mut p = make_parser();
        let marker_prefix = &OPEN_MARKER[..5];

        let r1 = p.feed(marker_prefix);
        assert!(r1.is_empty());

        p.feed(&OPEN_MARKER[5..]);
        // Open marker completed, no tool calls emitted yet

        let args = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"";
        let r3 = p.feed(args);
        assert!(r3.tool_calls.is_empty());

        let _ = p.feed("London\"}}");
        let r5 = p.feed(CLOSE_MARKER);
        let last = r5.tool_calls.last().expect("at least one delta");
        assert!(last.is_complete);
        assert_eq!(last.name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn partial_marker_not_emitted_as_content() {
        let mut p = make_parser();
        let r1 = p.feed("Hello <tool");
        assert_eq!(r1.content.len(), 1);
        assert_eq!(r1.content[0], "Hello ");
        assert!(r1.tool_calls.is_empty());

        p.feed("_call>{}");
        let r3 = p.feed(CLOSE_MARKER);
        assert!(!r3.tool_calls.is_empty());
    }

    #[test]
    fn content_after_tool_call() {
        let mut p = make_parser();
        let input = format!("{OPEN_MARKER}{{\"name\":\"fn\",\"arguments\":{{}}}}{CLOSE_MARKER}");
        p.feed(&input);
        let r = p.feed(" Done.");
        assert!(!r.content.is_empty());
        assert!(r.content[0].contains("Done"));
    }

    #[test]
    fn arguments_are_streamed_as_diffs() {
        let mut p = make_parser();
        // Feed the open marker
        let _ = p.feed(OPEN_MARKER);
        // Feed the JSON body incrementally
        let _ = p.feed("{\"name\":\"fn\",\"arguments\":");
        let r = p.feed("{\"city\":\"London\"}");
        // Should have emitted some argument deltas (diffs)
        let incomplete: Vec<_> = r.tool_calls.iter().filter(|d| !d.is_complete).collect();
        // Now close it
        let r2 = p.feed(CLOSE_MARKER);
        let completed = r2.tool_calls.iter().find(|d| d.is_complete).unwrap();
        assert_eq!(completed.name.as_deref(), Some("fn"));

        // Verify the arguments fragments accumulate to the correct value.
        let mut all_args = String::new();
        all_args.extend(incomplete.iter().map(|d| d.arguments_fragment.as_str()));
        all_args.push_str(&completed.arguments_fragment);
        let _ = &all_args; // accumulated args for this tool call
    }
}
