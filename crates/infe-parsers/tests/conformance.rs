//! Conformance runner for infe-parsers.
//!
//! Loads JSON fixtures from `conformance/fixtures/parsers/` and feeds
//! them through a `StreamingParser`, asserting the accumulated output
//! matches the expected deltas.
//!
//! Run with: `cargo test --test conformance`

use infe_parsers::{DialectRegistry, StreamingParser};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Fixture {
    name: String,
    dialect: String,
    #[serde(default)]
    source: String,
    #[serde(default)]
    description: String,
    input_chunks: Vec<String>,
    #[serde(default)]
    expected_tool_calls: Vec<ExpectedToolCall>,
    #[serde(default)]
    expected_reasoning: Vec<ExpectedReasoning>,
    #[serde(default)]
    expected_content: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ExpectedToolCall {
    name: Option<String>,
    arguments: Option<String>,
    #[serde(default)]
    is_complete: bool,
}

#[derive(Debug, Deserialize)]
struct ExpectedReasoning {
    #[serde(default)]
    fragment: String,
    #[serde(default)]
    is_complete: bool,
}

fn load_fixtures() -> Vec<Fixture> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("conformance")
        .join("fixtures")
        .join("parsers");

    let mut fixtures = Vec::new();

    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                let content = fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
                let fixture: Fixture = serde_json::from_str(&content)
                    .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));
                fixtures.push(fixture);
            }
        }
    }

    fixtures
}

fn run_fixture(fixture: &Fixture) {
    let dialect = DialectRegistry::create(&fixture.dialect).unwrap_or_else(|e| {
        panic!(
            "unknown dialect '{}' for {}: {e}",
            fixture.dialect, fixture.name
        )
    });

    let mut parser = StreamingParser::new(dialect);

    // Accumulate deltas across all feed calls and the final finish.
    let mut all_tool_calls = Vec::new();
    let mut all_reasoning = Vec::new();
    let mut all_content = Vec::new();

    for chunk in &fixture.input_chunks {
        let r = parser.feed(chunk);
        all_tool_calls.extend(r.tool_calls);
        all_reasoning.extend(r.reasoning);
        all_content.extend(r.content);
    }
    let r = parser.finish();
    all_tool_calls.extend(r.tool_calls);
    all_reasoning.extend(r.reasoning);
    all_content.extend(r.content);

    // Check tool calls: only count completed ones
    let completed: Vec<_> = all_tool_calls.iter().filter(|t| t.is_complete).collect();
    assert_eq!(
        completed.len(),
        fixture.expected_tool_calls.len(),
        "{}: expected {} completed tool calls, got {}",
        fixture.name,
        fixture.expected_tool_calls.len(),
        completed.len()
    );

    for (i, expected) in fixture.expected_tool_calls.iter().enumerate() {
        let actual = completed[i];
        if let Some(ref name) = expected.name {
            assert_eq!(
                actual.name.as_deref(),
                Some(name.as_str()),
                "{}: tool call {} name mismatch",
                fixture.name,
                i
            );
        }
        if let Some(ref args) = expected.arguments {
            assert_eq!(
                actual.arguments_fragment, *args,
                "{}: tool call {} arguments mismatch",
                fixture.name, i
            );
        }
    }

    // Check reasoning
    if !fixture.expected_reasoning.is_empty() {
        let actual_reasoning: Vec<_> = all_reasoning.iter().collect();
        assert_eq!(
            actual_reasoning.len(),
            fixture.expected_reasoning.len(),
            "{}: expected {} reasoning deltas, got {}",
            fixture.name,
            fixture.expected_reasoning.len(),
            actual_reasoning.len()
        );
        for (i, expected) in fixture.expected_reasoning.iter().enumerate() {
            assert_eq!(
                actual_reasoning[i].fragment, expected.fragment,
                "{}: reasoning {} fragment mismatch",
                fixture.name, i
            );
            assert_eq!(
                actual_reasoning[i].is_complete, expected.is_complete,
                "{}: reasoning {} is_complete mismatch",
                fixture.name, i
            );
        }
    }

    // Check content
    if !fixture.expected_content.is_empty() {
        let joined: String = all_content.join("");
        for expected in &fixture.expected_content {
            assert!(
                joined.contains(expected.as_str()),
                "{}: expected content '{}' not found in '{}'",
                fixture.name,
                expected,
                joined
            );
        }
    }
}

#[test]
fn conformance_all_fixtures() {
    let fixtures = load_fixtures();
    assert!(!fixtures.is_empty(), "no conformance fixtures found");

    for fixture in &fixtures {
        run_fixture(fixture);
    }
}
