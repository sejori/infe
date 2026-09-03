//! Error types for infe components.
//!
//! Every component returns [`ComponentResult`], which is `Result<T,
//! ComponentError>`. Errors are structured so that the shim layer can map them
//! to the appropriate engine exception (vLLM `SchedulerError`, `SGLang`
//! `SchedulingError`, etc.) without string parsing.

use thiserror::Error;

/// The result type returned by all infe component operations.
pub type ComponentResult<T> = Result<T, ComponentError>;

/// A structured error from an infe component.
///
/// The shim layer maps these to engine-specific exceptions. The variants are
/// deliberately coarse — the engine cares about "is this a bug or is this
/// load", not the internal details.
#[derive(Error, Debug)]
pub enum ComponentError {
    /// The input buffers or metadata are invalid for this step.
    ///
    /// Maps to a 400-class error in the engine. This is a bug in the shim or
    /// the engine's adapter code, not a transient failure.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The component's internal state is inconsistent.
    ///
    /// This is a bug in the component itself. The engine should fall back to
    /// stock and report the error.
    #[error("internal error in {component}: {detail}")]
    Internal {
        /// Which component produced the error (e.g. "infe-kv").
        component: &'static str,
        /// Human-readable detail.
        detail: String,
    },

    /// A buffer was too small for the step's output.
    ///
    /// The engine pre-allocates output buffers based on its own size estimates.
    /// If the component needs more space, it returns this error so the engine
    /// can reallocate and retry the step.
    #[error("buffer too small: needed {needed} bytes, have {have}")]
    BufferTooSmall {
        /// Bytes needed.
        needed: usize,
        /// Bytes available in the output buffer.
        have: usize,
    },

    /// A feature is not supported by this component at this engine version.
    ///
    /// The engine should fall back to stock for this feature.
    #[error("unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// The component's manifest declares this engine version as unsupported.
    #[error("engine version {version} not supported by {component} (requires {range})")]
    UnsupportedEngine {
        /// Component name.
        component: &'static str,
        /// The engine version that was attempted.
        version: String,
        /// The supported semver range from the manifest.
        range: String,
    },
}

impl ComponentError {
    /// Convenience constructor for [`ComponentError::Internal`].
    #[must_use]
    pub fn internal(component: &'static str, detail: impl Into<String>) -> Self {
        Self::Internal {
            component,
            detail: detail.into(),
        }
    }

    /// Whether this error is transient (retryable) or permanent (fall back).
    ///
    /// Only [`ComponentError::BufferTooSmall`] is transient — the engine can
    /// reallocate and retry. All others indicate a bug or unsupported feature
    /// that requires falling back to the stock path.
    #[must_use]
    pub fn is_transient(&self) -> bool {
        matches!(self, Self::BufferTooSmall { .. })
    }

    /// Whether this error should cause the engine to fall back to stock.
    #[must_use]
    pub fn requires_fallback(&self) -> bool {
        !self.is_transient()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_convenience() {
        let e = ComponentError::internal("infe-kv", "block table overflow");
        assert!(matches!(
            e,
            ComponentError::Internal {
                component: "infe-kv",
                ..
            }
        ));
    }

    #[test]
    fn buffer_too_small_is_transient() {
        let e = ComponentError::BufferTooSmall {
            needed: 1024,
            have: 512,
        };
        assert!(e.is_transient());
        assert!(!e.requires_fallback());
    }

    #[test]
    fn invalid_input_is_not_transient() {
        let e = ComponentError::InvalidInput("bad shape".into());
        assert!(!e.is_transient());
        assert!(e.requires_fallback());
    }

    #[test]
    fn unsupported_feature_requires_fallback() {
        let e = ComponentError::UnsupportedFeature("spec-decode v3".into());
        assert!(e.requires_fallback());
    }

    #[test]
    fn unsupported_engine_requires_fallback() {
        let e = ComponentError::UnsupportedEngine {
            component: "infe-parsers",
            version: "0.13.0".into(),
            range: ">=0.10, <0.13".into(),
        };
        assert!(e.requires_fallback());
    }
}
