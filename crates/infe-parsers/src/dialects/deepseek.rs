//! `DeepSeek` reasoning parser.
//!
//! DeepSeek-R1 and similar thinking models emit reasoning wrapped in
//! <think>...</think> tags. This parser extracts reasoning content
//! and emits `ReasoningDelta` fragments.

#![allow(clippy::cast_possible_truncation)]

use crate::parser::{DialectParser, ParseResult};
use crate::types::{ReasoningDelta, ToolCallArgState};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Debug, Default)]
pub struct DeepSeekReasoningParser {
    in_think: bool,
    pending: String,
}

impl DeepSeekReasoningParser {
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
}

impl DialectParser for DeepSeekReasoningParser {
    fn name(&self) -> &'static str {
        "deepseek_reasoning"
    }

    fn feed(&mut self, text: &str, _state: &mut ToolCallArgState, result: &mut ParseResult) {
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
            if self.in_think {
                // Search for the close marker.
                if let Some(pos) = remaining.find(THINK_CLOSE) {
                    // Emit content before the close marker.
                    if pos > 0 {
                        let chunk = &remaining[..pos];
                        result.reasoning.push(ReasoningDelta {
                            fragment: chunk.to_string(),
                            is_complete: false,
                        });
                    }
                    // Emit the completion delta.
                    result.reasoning.push(ReasoningDelta {
                        fragment: String::new(),
                        is_complete: true,
                    });
                    self.in_think = false;
                    remaining = &remaining[pos + THINK_CLOSE.len()..];
                } else if Self::is_partial_prefix(remaining, THINK_CLOSE) {
                    self.pending.push_str(remaining);
                    break;
                } else {
                    // Check if the tail is a partial prefix.
                    let split_pos = Self::find_tail_partial(remaining, THINK_CLOSE);
                    if split_pos > 0 {
                        let chunk = &remaining[..split_pos];
                        result.reasoning.push(ReasoningDelta {
                            fragment: chunk.to_string(),
                            is_complete: false,
                        });
                    }
                    if split_pos < remaining.len() {
                        self.pending.push_str(&remaining[split_pos..]);
                    }
                    break;
                }
            } else {
                // Search for the open marker.
                if let Some(consumed) = Self::match_marker(remaining, THINK_OPEN) {
                    self.in_think = true;
                    remaining = &remaining[consumed..];
                } else if Self::is_partial_prefix(remaining, THINK_OPEN) {
                    self.pending.push_str(remaining);
                    break;
                } else {
                    // Search for open marker in text.
                    if let Some(pos) = remaining.find(THINK_OPEN) {
                        if pos > 0 {
                            result.content.push(remaining[..pos].to_string());
                        }
                        remaining = &remaining[pos..];
                    } else {
                        // Check if tail is a partial prefix.
                        let split_pos = Self::find_tail_partial(remaining, THINK_OPEN);
                        if split_pos > 0 {
                            result.content.push(remaining[..split_pos].to_string());
                        }
                        if split_pos < remaining.len() {
                            self.pending.push_str(&remaining[split_pos..]);
                        }
                        break;
                    }
                }
            }
        }
    }

    fn reset(&mut self, state: &mut ToolCallArgState) {
        *state = ToolCallArgState::new();
        self.in_think = false;
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::StreamingParser;
    use crate::registry::DialectRegistry;

    fn make_parser() -> StreamingParser {
        let dialect = DialectRegistry::create("deepseek_reasoning").unwrap();
        StreamingParser::new(dialect)
    }

    #[test]
    fn plain_content_no_reasoning() {
        let mut p = make_parser();
        let r = p.feed("Hello, world!");
        assert!(r.reasoning.is_empty());
        assert!(!r.content.is_empty());
    }

    #[test]
    fn reasoning_single_block() {
        let mut p = make_parser();
        let input = format!("{THINK_OPEN}Let me think.{THINK_CLOSE}Done.");
        let r = p.feed(&input);
        assert!(!r.reasoning.is_empty());
        let last = r.reasoning.last().unwrap();
        assert!(last.is_complete);
        let joined: String = r.content.join("");
        assert!(joined.contains("Done"));
        assert!(!r.content.is_empty());
    }

    #[test]
    fn reasoning_streamed_across_chunks() {
        let mut p = make_parser();
        let r1 = p.feed("<thi");
        assert!(r1.is_empty());
        let r2 = p.feed("nk>");
        assert!(r2.reasoning.is_empty());
        let r3 = p.feed("Reasoning here");
        assert!(!r3.reasoning.is_empty());
        assert!(!r3.reasoning[0].is_complete);
        let r4 = p.feed(THINK_CLOSE);
        assert!(!r4.reasoning.is_empty());
        let last = r4.reasoning.last().unwrap();
        assert!(last.is_complete);
    }
}
