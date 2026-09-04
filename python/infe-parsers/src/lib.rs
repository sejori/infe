//! Python bindings for infe-parsers.
//!
//! This module exposes the streaming parser to Python via PyO3,
//! following the brief's boundary rule (§5.1): one call per engine step,
//! arrays in, arrays out. The parser is created per-request, fed batches
//! of decoded text chunks, and produces structured deltas that the engine
//! shim translates into its own protocol types.

#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::redundant_closure)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use infe_parsers::DialectRegistry;
use infe_parsers::StreamingParser;

/// A Python-facing streaming parser session.
///
/// Create one per request, feed decoded text chunks, call finish() at
/// end-of-stream. The engine shim wraps this and translates the returned
/// dicts into its own delta types.
#[pyclass(name = "StreamingParser")]
struct PyStreamingParser {
    inner: StreamingParser,
}

#[pymethods]
impl PyStreamingParser {
    /// Create a new parser for the given dialect name.
    ///
    /// Available dialects: "hermes", "llama3_json", "deepseek_reasoning".
    #[new]
    fn new(dialect: String) -> PyResult<Self> {
        let dialect_parser = DialectRegistry::create(&dialect)
            .map_err(|e| PyValueError::new_err(format!("Unknown dialect '{dialect}': {e}")))?;
        Ok(Self {
            inner: StreamingParser::new(dialect_parser),
        })
    }

    /// Feed a chunk of decoded text. Returns a dict with keys:
    ///   "tool_calls": list of dicts with index, id, name, arguments_fragment, is_complete
    ///   "reasoning": list of dicts with fragment, is_complete
    ///   "content": list of strings
    ///
    /// This is the hot path — call once per engine step with a batch of
    /// decoded tokens, not once per token.
    fn feed(&mut self, text: String, py: Python<'_>) -> PyResult<PyObject> {
        let result = self.inner.feed(&text);
        Ok(parse_result_to_py(&result, py))
    }

    /// Feed a batch of text chunks in one call (step-granular).
    /// Same return format as feed().
    #[allow(clippy::redundant_closure)]
    fn feed_batch(&mut self, chunks: &Bound<'_, PyList>, py: Python<'_>) -> PyResult<PyObject> {
        let chunk_strs: Vec<String> = chunks
            .iter()
            .map(|c| -> PyResult<String> { c.extract::<String>() })
            .collect::<PyResult<Vec<String>>>()?;
        let chunk_refs: Vec<&str> = chunk_strs.iter().map(String::as_str).collect();
        let result = self.inner.feed_batch(&chunk_refs);
        Ok(parse_result_to_py(&result, py))
    }

    /// Signal end-of-stream and flush any buffered partial content.
    /// Same return format as feed().
    fn finish(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let result = self.inner.finish();
        Ok(parse_result_to_py(&result, py))
    }

    /// Reset the parser for a new request (reuse the same dialect).
    fn reset(&mut self) {
        self.inner.reset();
    }

    #[allow(clippy::unused_self)]
    fn __repr__(&self) -> String {
        "infe_parsers.StreamingParser".to_string()
    }
}

/// Convert a ParseResult into a Python dict.
fn parse_result_to_py(result: &infe_parsers::ParseResult, py: Python<'_>) -> PyObject {
    let dict = PyDict::new(py);

    // tool_calls
    let tc_list = PyList::empty(py);
    for tc in &result.tool_calls {
        let tc_dict = PyDict::new(py);
        tc_dict.set_item("index", tc.index).unwrap();
        tc_dict.set_item("id", &tc.id).unwrap();
        tc_dict.set_item("name", &tc.name).unwrap();
        tc_dict
            .set_item("arguments_fragment", &tc.arguments_fragment)
            .unwrap();
        tc_dict.set_item("is_complete", tc.is_complete).unwrap();
        tc_list.append(tc_dict).unwrap();
    }
    dict.set_item("tool_calls", tc_list).unwrap();

    // reasoning
    let r_list = PyList::empty(py);
    for r in &result.reasoning {
        let r_dict = PyDict::new(py);
        r_dict.set_item("fragment", &r.fragment).unwrap();
        r_dict.set_item("is_complete", r.is_complete).unwrap();
        r_list.append(r_dict).unwrap();
    }
    dict.set_item("reasoning", r_list).unwrap();

    // content
    let c_list = PyList::empty(py);
    for c in &result.content {
        c_list.append(c).unwrap();
    }
    dict.set_item("content", c_list).unwrap();

    dict.into()
}

/// List all available dialect names.
#[pyfunction]
fn list_dialects() -> Vec<&'static str> {
    DialectRegistry::names()
}

/// The Python module.
#[pymodule]
fn _infe_parsers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(list_dialects, m)?)?;
    m.add_class::<PyStreamingParser>()?;
    Ok(())
}
