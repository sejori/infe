//! # infe-core
//!
//! Core infrastructure for the infe component framework: step-granular contract
//! traits, buffer helpers for `PyO3` crossings, unified error types, and a
//! `StepTimer` for CPU-side step instrumentation.
//!
//! ## The boundary rule
//!
//! Every component contract is **one call per engine step**: arrays in, arrays
//! out. Rust releases the GIL for the duration of the call. Data crosses as
//! `DLPack` / Arrow / numpy views over pre-allocated buffers, never as Python
//! objects. This crate defines the types that make that boundary safe.
//!
//! See `BRIEF.md` §5.1 for the full rationale.

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub mod buffer;
pub mod error;
pub mod manifest;
pub mod step;
pub mod timer;

pub use buffer::{BufferView, BufferViewMut, DType};
pub use error::{ComponentError, ComponentResult};
pub use manifest::{ComponentManifest, EngineSupport, ManifestCapability};
pub use step::{StepContext, StepInput, StepOutput, StepPlan, StepRequest, StepResult};
pub use timer::StepTimer;
