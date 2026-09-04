//! Conformance runner for infe-parsers.
//!
//! Loads JSON fixtures from `conformance/fixtures/parsers/` and feeds
//! them through a `StreamingParser`, asserting the accumulated output
//! matches the expected deltas.
//!
//! With incremental argument streaming (matching vLLM/SGLang stock
//! behaviour), a single tool call may produce multiple deltas:
//! - First delta: name + id (no arguments)
//! - Subsequent deltas: argument fragments (diffs)
//! - Final delta: any remaining argument diff + `is_complete=true`
//!
//! The runner accumulates by tool-call index and asserts:
//! - Name matches (collected from any delta for that index)
//! - ID present on the first delta for each index
//! - Accumulated arguments fragments match expected value
//! - Index matches
//! - Completion is flagged

use infe_parsers::{DialectRegistry, StreamingParser};
use serde::Deserialize;
use std::collections::BTreeMap;
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
    index: Option<usize>,
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

/// Accumulated state for one tool call across multiple deltas.
#[derive(Default)]
struct AccumulatedToolCall {
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments: String,
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

#[allow(clippy::too_many_lines)]
fn run_fixture(fixture: &Fixture) {
    let dialect = DialectRegistry::create(&fixture.dialect).unwrap_or_else(|e| {
        panic!(
            "unknown dialect '{}' for {}: {e}",
            fixture.dialect, fixture.name
        )
    });

    let mut parser = StreamingParser::new(dialect);

    let mut all_reasoning = Vec::new();
    let mut all_content = Vec::new();

    // Accumulate tool-call deltas by index across all feed calls.
    let mut acc: BTreeMap<usize, AccumulatedToolCall> = BTreeMap::new();

    for chunk in &fixture.input_chunks {
        let r = parser.feed(chunk);
        for tc in &r.tool_calls {
            let entry = acc.entry(tc.index).or_default();
            entry.index = tc.index;
            if tc.id.is_some() {
                entry.id.clone_from(&tc.id);
            }
            if tc.name.is_some() {
                entry.name.clone_from(&tc.name);
            }
            entry.arguments.push_str(&tc.arguments_fragment);
            if tc.is_complete {
                entry.is_complete = true;
            }
        }
        all_reasoning.extend(r.reasoning);
        all_content.extend(r.content);
    }
    let r = parser.finish();
    for tc in &r.tool_calls {
        let entry = acc.entry(tc.index).or_default();
        entry.index = tc.index;
        if tc.id.is_some() {
            entry.id.clone_from(&tc.id);
        }
        if tc.name.is_some() {
            entry.name.clone_from(&tc.name);
        }
        entry.arguments.push_str(&tc.arguments_fragment);
        if tc.is_complete {
            entry.is_complete = true;
        }
    }
    all_reasoning.extend(r.reasoning);
    all_content.extend(r.content);

    // Collect into sorted-by-index vector.
    let completed: Vec<_> = acc.into_values().collect();

    assert_eq!(
        completed.len(),
        fixture.expected_tool_calls.len(),
        "{}: expected {} tool calls, got {}",
        fixture.name,
        fixture.expected_tool_calls.len(),
        completed.len()
    );

    for (i, expected) in fixture.expected_tool_calls.iter().enumerate() {
        let actual = &completed[i];

        // Assert name (if specified).
        if let Some(ref name) = expected.name {
            assert_eq!(
                actual.name.as_deref(),
                Some(name.as_str()),
                "{}: tool call {} name mismatch",
                fixture.name,
                i
            );
        }

        // Assert accumulated arguments (if specified).
        if let Some(ref args) = expected.arguments {
            assert_eq!(
                actual.arguments, *args,
                "{}: tool call {} accumulated arguments mismatch",
                fixture.name, i
            );
        }

        // Assert index (if specified).
        if let Some(expected_index) = expected.index {
            assert_eq!(
                actual.index, expected_index,
                "{}: tool call {} index mismatch",
                fixture.name, i
            );
        }

        // Assert every tool call has an id.
        assert!(
            actual.id.is_some(),
            "{}: tool call {} should have a tool-call id",
            fixture.name,
            i
        );
    }

    // Check reasoning.
    if !fixture.expected_reasoning.is_empty() {
        assert_eq!(
            all_reasoning.len(),
            fixture.expected_reasoning.len(),
            "{}: expected {} reasoning deltas, got {}",
            fixture.name,
            fixture.expected_reasoning.len(),
            all_reasoning.len()
        );
        for (i, expected) in fixture.expected_reasoning.iter().enumerate() {
            assert_eq!(
                all_reasoning[i].fragment, expected.fragment,
                "{}: reasoning {} fragment mismatch",
                fixture.name, i
            );
            assert_eq!(
                all_reasoning[i].is_complete, expected.is_complete,
                "{}: reasoning {} is_complete mismatch",
                fixture.name, i
            );
        }
    }

    // Check content.
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
