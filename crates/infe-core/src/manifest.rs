#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
//! Component manifest schema (v0).
//!
//! Every infe component ships a manifest that declares its contract version,
//! capabilities, supported engines and versions, and conformance suite
//! reference. The manifest is a YAML file at
//! `registry/<component>/manifest.yaml` and is the source of truth for what
//! the component does and where it works.
//!
//! This module provides the Rust types for parsing and validating manifests.
//! The YAML deserialization lives in the `infe-manifest` crate (to avoid a
//! serde dependency in core); here we define the data model and validation
//! logic.

use std::collections::BTreeMap;

/// The manifest schema version. Increment when the manifest format itself
/// changes (not when a component's version bumps).
pub const MANIFEST_SCHEMA_VERSION: u32 = 0;

/// A component manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentManifest {
    /// Schema version (currently 0).
    pub schema_version: u32,

    /// Component name (e.g. "infe-parsers", "infe-kv", "infe-sched").
    pub name: String,

    /// Component version (semver).
    pub version: String,

    /// Human-readable description.
    pub description: String,

    /// Contract version — the trait/API version the component implements.
    /// Components with the same contract version are interchangeable.
    pub contract_version: u32,

    /// What the component does (for the registry UI and manifest validation).
    pub capabilities: Vec<ManifestCapability>,

    /// Which engines and version ranges this component supports.
    pub engines: Vec<EngineSupport>,

    /// Reference to the conformance suite (path relative to repo root).
    pub conformance_ref: String,

    /// Features the component does NOT support (from the engine's stock
    /// feature set). The engine falls back to stock for these.
    pub unsupported_features: Vec<String>,

    /// Optional metadata (links, author, license, etc.).
    pub metadata: BTreeMap<String, String>,
}

/// A capability declared by a component.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ManifestCapability {
    /// Produces scheduling decisions (admit, defer, preempt).
    Scheduling,
    /// Manages KV cache blocks (allocation, prefix matching, eviction).
    KvManagement,
    /// Parses tool-call and reasoning streams.
    StreamParsing,
    /// Builds attention metadata (block tables, `cu_seqlens`, slot mappings).
    AttentionMetadata,
    /// Custom capability with a free-form name.
    Custom(String),
}

impl std::fmt::Display for ManifestCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scheduling => f.write_str("scheduling"),
            Self::KvManagement => f.write_str("kv-management"),
            Self::StreamParsing => f.write_str("stream-parsing"),
            Self::AttentionMetadata => f.write_str("attention-metadata"),
            Self::Custom(name) => write!(f, "custom:{name}"),
        }
    }
}

/// Engine support declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineSupport {
    /// Engine name: "vllm" or "sglang".
    pub engine: String,

    /// Semver range for supported versions (e.g. ">=0.10, <0.14").
    pub version_range: String,

    /// How the component is registered: the flag or hook name.
    pub registration: String,

    /// Whether the integration is verified or a documented gap.
    pub status: IntegrationStatus,
}

/// The status of an engine integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegrationStatus {
    /// The shim is implemented and conformance passes.
    Verified,
    /// The shim exists but is untested against the pinned engine version.
    Experimental,
    /// No seam exists in this engine; the component does not work here.
    DocumentedGap,
}

impl std::fmt::Display for IntegrationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Verified => f.write_str("verified"),
            Self::Experimental => f.write_str("experimental"),
            Self::DocumentedGap => f.write_str("documented-gap"),
        }
    }
}

impl ComponentManifest {
    /// Create a new manifest with the current schema version.
    #[must_use]
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            name: name.into(),
            version: version.into(),
            description: String::new(),
            contract_version: 0,
            capabilities: Vec::new(),
            engines: Vec::new(),
            conformance_ref: String::new(),
            unsupported_features: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Check if this manifest supports a given engine at a given version.
    ///
    /// This is a simple string-prefix check for now; full semver range parsing
    /// will be added when the first real manifest is written.
    #[must_use]
    pub fn supports_engine(&self, engine: &str, version: &str) -> bool {
        self.engines.iter().any(|e| {
            e.engine == engine
                && e.status != IntegrationStatus::DocumentedGap
                && version_in_range(version, &e.version_range)
        })
    }

    /// Validate the manifest for internal consistency.
    ///
    /// Returns a list of validation errors (empty if valid).
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            errors.push(format!(
                "schema_version is {}, expected {}",
                self.schema_version, MANIFEST_SCHEMA_VERSION
            ));
        }

        if self.name.is_empty() {
            errors.push("name is empty".to_string());
        }

        if !self.name.starts_with("infe-") {
            errors.push(format!("name '{}' does not start with 'infe-'", self.name));
        }

        if self.version.is_empty() {
            errors.push("version is empty".to_string());
        }

        if self.capabilities.is_empty() {
            errors.push("no capabilities declared".to_string());
        }

        if self.engines.is_empty() {
            errors.push("no engine support declared".to_string());
        }

        for e in &self.engines {
            if e.engine != "vllm" && e.engine != "sglang" {
                errors.push(format!(
                    "unknown engine '{}' (expected 'vllm' or 'sglang')",
                    e.engine
                ));
            }
            if e.version_range.is_empty() {
                errors.push(format!("engine '{}' has empty version_range", e.engine));
            }
        }

        errors
    }
}

/// Simple version-in-range check. Supports `>=X.Y` and `<X.Y` constraints
/// separated by commas. This is a placeholder for full semver parsing.
fn version_in_range(version: &str, range: &str) -> bool {
    let parse_ver = |s: &str| -> (u32, u32, u32) {
        let parts: Vec<u32> = s
            .split('.')
            .map(|p| p.trim().parse().unwrap_or(0))
            .collect();
        (
            parts.first().copied().unwrap_or(0),
            parts.second().copied().unwrap_or(0),
            parts.get(2).copied().unwrap_or(0),
        )
    };

    let v = parse_ver(version);

    for constraint in range.split(',') {
        let constraint = constraint.trim();
        if let Some(rest) = constraint.strip_prefix(">=") {
            let min = parse_ver(rest.trim());
            if v < min {
                return false;
            }
        } else if let Some(rest) = constraint.strip_prefix("<") {
            let max = parse_ver(rest.trim());
            if v >= max {
                return false;
            }
        }
    }
    true
}

// Helper trait for cleaner array access
trait SecondExt<T> {
    fn second(&self) -> Option<&T>;
}

impl<T> SecondExt<T> for [T] {
    fn second(&self) -> Option<&T> {
        self.get(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_manifest() -> ComponentManifest {
        ComponentManifest {
            schema_version: 0,
            name: "infe-parsers".to_string(),
            version: "0.1.0".to_string(),
            description: "Tool-call and reasoning stream parsers".to_string(),
            contract_version: 0,
            capabilities: vec![ManifestCapability::StreamParsing],
            engines: vec![EngineSupport {
                engine: "vllm".to_string(),
                version_range: ">=0.10, <0.14".to_string(),
                registration: "--tool-call-parser".to_string(),
                status: IntegrationStatus::Verified,
            }],
            conformance_ref: "conformance/parsers/".to_string(),
            unsupported_features: vec![],
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn manifest_valid() {
        let m = example_manifest();
        assert!(m.validate().is_empty(), "{:?}", m.validate());
    }

    #[test]
    fn manifest_invalid_name() {
        let mut m = example_manifest();
        m.name = "parsers".to_string(); // missing infe- prefix
        let errors = m.validate();
        assert!(errors.iter().any(|e| e.contains("does not start")));
    }

    #[test]
    fn manifest_no_capabilities() {
        let mut m = example_manifest();
        m.capabilities.clear();
        let errors = m.validate();
        assert!(errors.iter().any(|e| e.contains("no capabilities")));
    }

    #[test]
    fn supports_engine_in_range() {
        let m = example_manifest();
        assert!(m.supports_engine("vllm", "0.12.0"));
        assert!(m.supports_engine("vllm", "0.10.5"));
        assert!(m.supports_engine("vllm", "0.13.99"));
    }

    #[test]
    fn does_not_support_engine_out_of_range() {
        let m = example_manifest();
        assert!(!m.supports_engine("vllm", "0.9.0"));
        assert!(!m.supports_engine("vllm", "0.14.0"));
        assert!(!m.supports_engine("sglang", "0.5.0"));
    }

    #[test]
    fn does_not_support_documented_gap() {
        let m = ComponentManifest {
            engines: vec![EngineSupport {
                engine: "sglang".to_string(),
                version_range: ">=0.4".to_string(),
                registration: "N/A".to_string(),
                status: IntegrationStatus::DocumentedGap,
            }],
            ..example_manifest()
        };
        assert!(!m.supports_engine("sglang", "0.5.0"));
    }

    #[test]
    fn capability_display() {
        assert_eq!(ManifestCapability::Scheduling.to_string(), "scheduling");
        assert_eq!(
            ManifestCapability::KvManagement.to_string(),
            "kv-management"
        );
        assert_eq!(
            ManifestCapability::Custom("foo".to_string()).to_string(),
            "custom:foo"
        );
    }

    #[test]
    fn schema_version_constant() {
        assert_eq!(MANIFEST_SCHEMA_VERSION, 0);
    }
}
