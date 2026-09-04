#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]

use crate::parser::{DialectParser, ParseResult};
use crate::types::{ToolCallArgState, ToolCallDelta, ToolCallState};

const OPEN_MARKER: &str = "\u{3c}tool_call\u{3e}";
const CLOSE_MARKER: &str = "\u{3c}/tool_call\u{3e}";

/// Hermes tool-call parser.
///
/// Matches vLLM's `hermes_tool_parser.py` streaming semantics:
///
/// 1. The JSON body between `` and `` is accumulated incrementally.
/// 2. On every `feed`, we attempt to extract the `name` field via regex.
///    When the name is available and hasn't been sent yet, we emit a
///    delta with `name` + `id`.
/// 3. We extract the `arguments` value (everything after `"arguments":`) and
///    compute the diff against what was already streamed, emitting only the
///    new characters. This matches vLLM's `_compute_args_diff`.
/// 4. At completion (close marker), we emit any remaining argument diff and
///    mark the call complete.
///
/// Key behaviour: arguments are streamed **as they arrive**, not buffered
/// until completion. This is what clients expect — the SSE stream shows
/// arguments growing token by token.
#[derive(Debug, Default)]
pub struct HermesParser {
    /// Text pending analysis (partial markers that span chunk boundaries).
    pending: String,
    /// Whether we've started inside a tool-call block (used across feed calls).
    in_tool_call: bool,
    /// Buffer of text accumulated before the name was found — content that
    /// arrived before we could parse the `name` field. Emptied once the name
    /// is emitted and argument streaming begins.
    pre_name_buffer: String,
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
    /// Mirrors vLLM's `partial_tag_overlap`.
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

    /// Extract the tool name from a (possibly partial) JSON body.
    /// Mirrors vLLM's `_extract_tool_name`: regex `"name"\s*:\s*"([^"]+)"`.
    /// Returns None if the name field hasn't been completed yet.
    fn extract_name(json_str: &str) -> Option<String> {
        // Find "name" key
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

    /// Extract the raw arguments text — everything after `"arguments":`.
    /// Mirrors vLLM's `_extract_tool_args`. When `is_complete`, strips the
    /// trailing `}` that closes the outer wrapper object.
    ///
    /// Returns the raw argument string (a fragment of JSON like `{"city":"London"}`).
    /// Returns None if the `"arguments":` key hasn't been seen yet.
    fn extract_arguments_raw(json_str: &str, is_complete: bool) -> Option<String> {
        // Find "arguments" key
        let args_idx = json_str.find("\"arguments\"")?;
        let after_key = &json_str[args_idx + 11..];
        let colon_idx = after_key.find(':')?;
        let raw = &after_key[colon_idx + 1..];
        let mut raw = raw.trim_start().to_string();

        if is_complete {
            // Strip the trailing } that closes the wrapper object.
            raw = raw.trim_end().to_string();
            if raw.ends_with('}') {
                raw.truncate(raw.len() - 1);
                raw = raw.trim_end().to_string();
            }
        }
        Some(raw)
    }

    /// Check if the JSON string is a complete, parseable JSON object.
    /// Mirrors vLLM's `is_complete_json`.
    fn is_complete_json(json_str: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(json_str).is_ok()
    }

    /// Try to emit incremental deltas for the current tool-call state.
    /// This is called after every chunk of text is accumulated.
    fn try_emit_incremental(
        state: &mut ToolCallArgState,
        json_body: &str,
        result: &mut ParseResult,
    ) {
        let is_complete = Self::is_complete_json(json_body);

        // If we haven't sent the name yet, try to extract and emit it.
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
                // Can't do anything until the name is available.
                return;
            }
        }

        // Stream argument diffs.
        if let Some(args_raw) = Self::extract_arguments_raw(json_body, is_complete) {
            let already = state.last_emitted_args_len;
            if args_raw.len() > already {
                let diff = &args_raw[already..];
                state.last_emitted_args_len = args_raw.len();
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

#[allow(clippy::too_many_lines)]
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
                        state.begin_tool_call();
                        self.in_tool_call = true;
                        self.pre_name_buffer.clear();
                        remaining = &remaining[consumed..];
                    } else if Self::is_partial_prefix(remaining, OPEN_MARKER) {
                        self.pending.push_str(remaining);
                        break;
                    } else {
                        // Search for the open marker anywhere in the text.
                        if let Some(pos) = remaining.find('<') {
                            let slice = &remaining[pos..];
                            if let Some(consumed) = Self::match_marker(slice, OPEN_MARKER) {
                                if pos > 0 {
                                    result.content.push(remaining[..pos].to_string());
                                }
                                state.begin_tool_call();
                                self.in_tool_call = true;
                                self.pre_name_buffer.clear();
                                remaining = &remaining[pos + consumed..];
                                continue;
                            }
                            if Self::is_partial_prefix(slice, OPEN_MARKER) {
                                if pos > 0 {
                                    result.content.push(remaining[..pos].to_string());
                                }
                                self.pending.push_str(slice);
                                break;
                            }
                        }
                        // No marker found — emit all as content, but hold back
                        // any tail that could be a partial open marker.
                        let split_pos = Self::find_tail_partial(remaining, OPEN_MARKER);
                        result.content.push(remaining[..split_pos].to_string());
                        if split_pos < remaining.len() {
                            self.pending.push_str(&remaining[split_pos..]);
                        }
                        break;
                    }
                }
                ToolCallState::InToolCall => {
                    // Search for the close marker in remaining.
                    if let Some(pos) = remaining.find(CLOSE_MARKER) {
                        // Accumulate chunk before close marker.
                        let chunk = &remaining[..pos];
                        if !chunk.is_empty() {
                            state.arguments_buffer.push_str(chunk);
                        }

                        // Now the JSON body is complete — try to emit final
                        // incremental deltas + the completion delta.
                        let json_body = state.arguments_buffer.clone();
                        let _is_complete = Self::is_complete_json(&json_body);

                        // Emit any remaining name/args diff.
                        Self::try_emit_incremental(state, &json_body, result);

                        // Also compute the diff at completion: if the JSON was
                        // complete before the close marker arrived, the args
                        // may already be fully emitted. But if is_complete was
                        // false before (partial JSON), we need to emit the
                        // final args now with is_complete=true.
                        if let Some(args_raw) = Self::extract_arguments_raw(&json_body, true) {
                            let already = state.last_emitted_args_len;
                            if args_raw.len() > already {
                                let diff = &args_raw[already..];
                                result.tool_calls.push(ToolCallDelta {
                                    index: state.index.unwrap_or(0),
                                    id: None,
                                    name: None,
                                    arguments_fragment: diff.to_string(),
                                    is_complete: true,
                                });
                                state.last_emitted_args_len = args_raw.len();
                            } else {
                                // Args were already fully streamed — emit a
                                // bare completion delta.
                                result.tool_calls.push(ToolCallDelta {
                                    index: state.index.unwrap_or(0),
                                    id: None,
                                    name: None,
                                    arguments_fragment: String::new(),
                                    is_complete: true,
                                });
                            }
                        } else {
                            // No arguments key — emit completion with empty args.
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
                        self.in_tool_call = false;
                        self.pre_name_buffer.clear();
                        remaining = &remaining[pos + CLOSE_MARKER.len()..];
                    } else if Self::is_partial_prefix(remaining, CLOSE_MARKER) {
                        self.pending.push_str(remaining);
                        break;
                    } else {
                        // No close marker found. Accumulate the raw text and
                        // try to emit incremental argument diffs.
                        // Hold back any tail that could be a partial close marker.
                        let split_pos = Self::find_tail_partial(remaining, CLOSE_MARKER);
                        let chunk = &remaining[..split_pos];
                        if chunk.is_empty() {
                        } else {
                            state.arguments_buffer.push_str(chunk);
                            // Try incremental emission now.
                            let buf = state.arguments_buffer.clone();
                            Self::try_emit_incremental(state, &buf, result);
                        }
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
        self.in_tool_call = false;
        self.pre_name_buffer.clear();
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

    /// Accumulate all arguments_fragment values from a list of deltas.
    fn accumulate_args(deltas: &[ToolCallDelta]) -> String {
        deltas
            .iter()
            .map(|d| d.arguments_fragment.as_str())
            .collect()
    }

    /// Find the name from the first delta that carries it.
    fn find_name(deltas: &[ToolCallDelta]) -> Option<&str> {
        deltas.iter().find_map(|d| d.name.as_deref())
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
        assert_eq!(find_name(&r.tool_calls).unwrap(), "fn");
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
        assert_eq!(find_name(&r.tool_calls).unwrap(), "get_weather");
        let args = accumulate_args(&r.tool_calls);
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
        let names: Vec<_> = r
            .tool_calls
            .iter()
            .filter_map(|d| d.name.as_deref())
            .collect();
        assert!(names.contains(&"fn1"));
        assert!(names.contains(&"fn2"));
    }

    #[test]
    fn tool_call_streamed_across_chunks() {
        let mut p = make_parser();
        let marker_prefix = &OPEN_MARKER[..5];
        let r1 = p.feed(marker_prefix);
        assert!(r1.is_empty());
        p.feed(&OPEN_MARKER[5..]);
        let args = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"";
        let r3 = p.feed(args);
        assert!(
            !r3.tool_calls.is_empty(),
            "should emit name delta when name is available"
        );
        assert_eq!(find_name(&r3.tool_calls).unwrap(), "get_weather");
        assert!(r3.tool_calls[0].id.is_some());
        let _ = p.feed("London\"}}");
        let r5 = p.feed(CLOSE_MARKER);
        let last = r5.tool_calls.last().expect("at least one delta");
        assert!(last.is_complete);
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
        let _ = p.feed(OPEN_MARKER);
        let r_name = p.feed("{\"name\":\"fn\",\"arguments\":");
        // Name should be emitted in this delta (when it becomes available)
        assert_eq!(find_name(&r_name.tool_calls).unwrap(), "fn");
        let r = p.feed("{\"city\":\"London\"}");
        assert!(!r.tool_calls.is_empty());
        let r2 = p.feed(CLOSE_MARKER);
        let _completed = r2
            .tool_calls
            .iter()
            .find(|d| d.is_complete)
            .expect("a completion delta");
    }

    #[test]
    fn incremental_argument_streaming() {
        let mut p = make_parser();
        let _ = p.feed(OPEN_MARKER);
        let _ = p.feed("{\"name\":\"fn\",\"arguments\":");
        let mut all_args = String::new();
        for piece in ["{\"city\":", "\"London\"}"] {
            let r = p.feed(piece);
            for tc in &r.tool_calls {
                all_args.push_str(&tc.arguments_fragment);
            }
        }
        let r = p.feed(CLOSE_MARKER);
        for tc in &r.tool_calls {
            all_args.push_str(&tc.arguments_fragment);
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
