//! Dialect registry — build-time registration of all available parser
//! dialects.
//!
//! Dialects are selected by name at engine start. The registry is a simple
//! match on a string — no dynamic loading, no plugin discovery. New dialects
//! are added by extending the match arms in [`DialectRegistry::create`].
//!
//! Each dialect is either a **tool-call** parser (registered through the
//! engine's tool-parser interface) or a **reasoning** parser (registered
//! through the reasoning-parser interface). The [`ParserKind`] tag tells the
//! shims which engine interface to use.

use crate::dialects;
use crate::parser::DialectParser;
use crate::types::{ParseError, ParserKind};

/// A registry of available parser dialects.
#[derive(Debug, Clone, Default)]
pub struct DialectRegistry;

impl DialectRegistry {
    /// List all registered dialect names.
    #[must_use]
    pub fn names() -> Vec<&'static str> {
        vec![
            dialects::HermesParser::new().name(),
            dialects::Llama3JsonParser::new().name(),
            dialects::DeepSeekReasoningParser::new().name(),
        ]
    }

    /// List only tool-call dialect names.
    #[must_use]
    pub fn tool_dialects() -> Vec<&'static str> {
        vec!["hermes", "llama3_json"]
    }

    /// List only reasoning dialect names.
    #[must_use]
    pub fn reasoning_dialects() -> Vec<&'static str> {
        vec!["deepseek_reasoning"]
    }

    /// Get the kind (tool or reasoning) for a dialect name.
    #[must_use]
    pub fn kind(name: &str) -> Option<ParserKind> {
        match name {
            "hermes" | "llama3_json" => Some(ParserKind::Tool),
            "deepseek_reasoning" => Some(ParserKind::Reasoning),
            _ => None,
        }
    }

    /// Create a new dialect parser by name.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError::UnknownDialect`] if the name is not recognised.
    pub fn create(name: &str) -> Result<Box<dyn DialectParser>, ParseError> {
        match name {
            "hermes" => Ok(Box::new(dialects::HermesParser::new())),
            "llama3_json" => Ok(Box::new(dialects::Llama3JsonParser::new())),
            "deepseek_reasoning" => Ok(Box::new(dialects::DeepSeekReasoningParser::new())),
            other => Err(ParseError::UnknownDialect(other.to_string())),
        }
    }

    /// Check if a dialect is registered.
    #[must_use]
    pub fn contains(name: &str) -> bool {
        matches!(name, "hermes" | "llama3_json" | "deepseek_reasoning")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lists_known_dialects() {
        let names = DialectRegistry::names();
        assert!(names.contains(&"hermes"));
        assert!(names.contains(&"llama3_json"));
        assert!(names.contains(&"deepseek_reasoning"));
    }

    #[test]
    fn registry_tool_dialects() {
        let tools = DialectRegistry::tool_dialects();
        assert_eq!(tools, vec!["hermes", "llama3_json"]);
    }

    #[test]
    fn registry_reasoning_dialects() {
        let reasoning = DialectRegistry::reasoning_dialects();
        assert_eq!(reasoning, vec!["deepseek_reasoning"]);
    }

    #[test]
    fn registry_kind_lookup() {
        assert_eq!(DialectRegistry::kind("hermes"), Some(ParserKind::Tool));
        assert_eq!(
            DialectRegistry::kind("deepseek_reasoning"),
            Some(ParserKind::Reasoning)
        );
        assert_eq!(DialectRegistry::kind("nonexistent"), None);
    }

    #[test]
    fn registry_creates_hermes() {
        let p = DialectRegistry::create("hermes");
        assert!(p.is_ok());
    }

    #[test]
    fn registry_unknown_dialect_errors() {
        let p = DialectRegistry::create("nonexistent");
        assert!(matches!(p, Err(ParseError::UnknownDialect(_))));
    }
}
