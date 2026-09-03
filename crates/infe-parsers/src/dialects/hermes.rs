#![allow(clippy::cast_possible_truncation)]

use crate::parser::{DialectParser, ParseResult};
use crate::types::{ToolCallArgState, ToolCallDelta, ToolCallState};

fn open_marker() -> &'static str {
    "<tool_call>"
}
fn close_marker() -> &'static str {
    "</tool_call>"
}

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

    fn extract_name(json_str: &str) -> (Option<String>, String) {
        if let Some(name_idx) = json_str.find("\"name\"") {
            let after_key = json_str[name_idx + 6..].trim_start();
            if let Some(colon_idx) = after_key.find(':') {
                let after_colon = after_key[colon_idx + 1..].trim_start();
                if let Some(rest) = after_colon.strip_prefix('"') {
                    if let Some(end) = rest.find('"') {
                        let name = rest[..end].to_string();
                        return (Some(name), json_str.to_string());
                    }
                }
            }
        }
        (None, json_str.to_string())
    }

    /// Find the position of a partial marker prefix at the END of text.
    /// Only matches at the tail, not anywhere in the middle.
    fn find_tail_partial(text: &str, marker: &str) -> usize {
        // Check progressively shorter suffixes of text.
        let max_len = text.len().min(marker.len());
        for len in (1..=max_len).rev() {
            let suffix = &text[text.len() - len..];
            if marker.strip_prefix(suffix).is_some() {
                return text.len() - len;
            }
        }
        text.len()
    }
}

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
        let open = open_marker();
        let close = close_marker();

        while !remaining.is_empty() {
            match state.state {
                ToolCallState::Idle => {
                    if let Some(consumed) = Self::match_marker(remaining, open) {
                        state.state = ToolCallState::InToolCall;
                        remaining = &remaining[consumed..];
                    } else if Self::is_partial_prefix(remaining, open) {
                        self.pending.push_str(remaining);
                        break;
                    } else {
                        // Search for the open marker anywhere in the text.
                        if let Some(pos) = remaining.find('<') {
                            let slice = &remaining[pos..];
                            if Self::is_partial_prefix(slice, open) {
                                // Emit content before the partial marker.
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
                    if let Some(pos) = remaining.find(close) {
                        // Emit content before the close marker.
                        let chunk = &remaining[..pos];
                        if !chunk.is_empty() {
                            state.arguments_buffer.push_str(chunk);
                            let delta = ToolCallDelta {
                                index: state.index.unwrap_or(0),
                                id: state.id.take(),
                                name: None,
                                arguments_fragment: chunk.to_string(),
                                is_complete: false,
                            };
                            result.tool_calls.push(delta);
                        }
                        // Process the close marker.
                        let (name, arguments) = Self::extract_name(&state.arguments_buffer);
                        let delta = ToolCallDelta {
                            index: state.index.unwrap_or(0),
                            id: state.id.take(),
                            name,
                            arguments_fragment: arguments,
                            is_complete: true,
                        };
                        result.tool_calls.push(delta);
                        state.state = ToolCallState::Complete;
                        state.arguments_buffer.clear();
                        state.name_buffer.clear();
                        remaining = &remaining[pos + close.len()..];
                    } else if Self::is_partial_prefix(remaining, close) {
                        self.pending.push_str(remaining);
                        break;
                    } else {
                        // Check if the tail is a partial prefix of close.
                        let split_pos = Self::find_tail_partial(remaining, close);
                        let chunk = &remaining[..split_pos];
                        if !chunk.is_empty() {
                            state.arguments_buffer.push_str(chunk);
                            let delta = ToolCallDelta {
                                index: state.index.unwrap_or(0),
                                id: state.id.take(),
                                name: None,
                                arguments_fragment: chunk.to_string(),
                                is_complete: false,
                            };
                            result.tool_calls.push(delta);
                        }
                        if split_pos < remaining.len() {
                            self.pending.push_str(&remaining[split_pos..]);
                        }
                        break;
                    }
                }
                ToolCallState::InArguments | ToolCallState::Complete => {
                    state.state = ToolCallState::Idle;
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
        let input = format!(
            "{}{{\"name\":\"fn\",\"arguments\":{{}}}}{}",
            "<tool_call>", "</tool_call>"
        );
        let r = p.feed(&input);
        let completed: Vec<_> = r.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].name.as_deref(), Some("fn"));
    }

    #[test]
    fn tool_call_streamed_across_chunks() {
        let mut p = make_parser();

        let r1 = p.feed("<tool_");
        assert!(r1.is_empty());

        let r2 = p.feed("call>");
        assert!(r2.tool_calls.is_empty());

        let args = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"";
        let r3 = p.feed(args);
        assert!(!r3.tool_calls.is_empty());

        let _ = p.feed("London\"}}");
        let r5 = p.feed("</tool_call>");
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

        let r2 = p.feed("_call>{}");
        // r2 contains the arguments fragment, but not a completed tool call
        let completed: Vec<_> = r2.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert!(completed.is_empty());
        let r3 = p.feed("</tool_call>");
        assert!(!r3.tool_calls.is_empty());
    }

    #[test]
    fn content_after_tool_call() {
        let mut p = make_parser();
        let input = format!(
            "{}{{\"name\":\"fn\",\"arguments\":{{}}}}{}",
            "<tool_call>", "</tool_call>"
        );
        p.feed(&input);
        let r = p.feed(" Done.");
        assert!(!r.content.is_empty());
        assert!(r.content[0].contains("Done"));
    }

    #[test]
    fn multiple_tool_calls() {
        let mut p = make_parser();
        let input = format!(
            "{}{{\"name\":\"fn1\",\"arguments\":{{\"a\":1}}}}{}{}{{\"name\":\"fn2\",\"arguments\":{{\"b\":2}}}}{}",
            "<tool_call>", "</tool_call>", "<tool_call>", "</tool_call>"
        );
        let r = p.feed(&input);
        let completed: Vec<_> = r.tool_calls.iter().filter(|d| d.is_complete).collect();
        assert_eq!(completed.len(), 2);
        assert_eq!(completed[0].name.as_deref(), Some("fn1"));
        assert_eq!(completed[1].name.as_deref(), Some("fn2"));
    }
}
