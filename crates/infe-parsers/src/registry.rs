//! Dialect registry — build-time registration of all available parser
//! dialects.
//!
//! Dialects are selected by name at engine start. The registry is a simple
//! match on a string — no dynamic loading, no plugin discovery. New dialects
//! are added by extending the match arms in [`DialectRegistry::create`].

use crate::dialects;
use crate::parser::DialectParser;
use crate::types::ParseError;

/// A registry of available parser dialects.
///
/// This is a compile-time registry — all dialects are linked in, and the
/// engine selects one by name. This avoids the complexity of dynamic plugin
/// loading while keeping the door open for a future dynamic registry.
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
