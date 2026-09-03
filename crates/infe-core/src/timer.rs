//! `StepTimer` — CPU-side step instrumentation for infe components.
//!
//! Every component wraps its `step` call in a `StepTimer` so that CPU-side
//! step time is reported identically across all components and engines. The
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
//! timer measures wall-clock time from the entry to the Rust call to the
//! return, excluding any time the engine spent gathering inputs or processing
//! outputs on the Python side.
//!
//! The timer is designed to have near-zero overhead when disabled: it checks
//! a single `AtomicBool` on the hot path.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A timer for measuring CPU-side step time.
///
/// # Usage
///
/// ```ignore
/// let mut timer = StepTimer::new("infe-kv");
/// // In the step call:
/// let _guard = timer.start(ctx.step_id);
/// // ... do work ...
/// // guard drops here, recording the elapsed time
/// ```
///
/// # Overhead
///
/// When `enabled` is `false` (the default), `start` returns immediately
/// without allocating or touching a clock. When enabled, it takes one
/// `Instant::now()` call on entry and one on drop.
pub struct StepTimer {
    /// Component name for labelling.
    name: &'static str,

    /// Whether timing is enabled.
    enabled: AtomicBool,

    /// Total number of steps timed.
    steps: AtomicU64,

    /// Total elapsed time in nanoseconds.
    total_ns: AtomicU64,

    /// Minimum step time in nanoseconds.
    min_ns: AtomicU64,

    /// Maximum step time in nanoseconds.
    max_ns: AtomicU64,

    /// Last step time in nanoseconds (for p99 estimation).
    last_ns: AtomicU64,
}

impl StepTimer {
    /// Create a new timer for the named component. Timing is disabled by
    /// default.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            enabled: AtomicBool::new(false),
            steps: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
            min_ns: AtomicU64::new(u64::MAX),
            max_ns: AtomicU64::new(0),
            last_ns: AtomicU64::new(0),
        }
    }

    /// Enable or disable timing.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Whether timing is currently enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Start timing a step. Returns a guard that records the elapsed time
    /// when dropped. If timing is disabled, returns a no-op guard.
    #[must_use]
    pub fn start(&self, step_id: u64) -> StepGuard<'_> {
        if self.is_enabled() {
            StepGuard {
                timer: self,
                start: Instant::now(),
                step_id: Some(step_id),
            }
        } else {
            StepGuard {
                timer: self,
                start: Instant::now(),
                step_id: None,
            }
        }
    }

    /// Record a completed step's duration (used internally by the guard).
    fn record(&self, duration: Duration) {
        let ns = duration.as_nanos() as u64;
        self.last_ns.store(ns, Ordering::Relaxed);
        self.total_ns.fetch_add(ns, Ordering::Relaxed);
        self.steps.fetch_add(1, Ordering::Relaxed);

        // Update min/max with compare-and-swap loops.
        let mut current_min = self.min_ns.load(Ordering::Relaxed);
        while ns < current_min {
            match self.min_ns.compare_exchange_weak(
                current_min,
                ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }

        let mut current_max = self.max_ns.load(Ordering::Relaxed);
        while ns > current_max {
            match self.max_ns.compare_exchange_weak(
                current_max,
                ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    /// Number of steps recorded.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }

    /// Mean step time in nanoseconds (0 if no steps recorded).
    #[must_use]
    pub fn mean_ns(&self) -> u64 {
        let steps = self.steps();
        if steps == 0 {
            return 0;
        }
        self.total_ns() / steps
    }

    /// Total elapsed time in nanoseconds.
    #[must_use]
    pub fn total_ns(&self) -> u64 {
        self.total_ns.load(Ordering::Relaxed)
    }

    /// Minimum step time in nanoseconds (0 if no steps recorded).
    #[must_use]
    pub fn min_ns(&self) -> u64 {
        let v = self.min_ns.load(Ordering::Relaxed);
        if v == u64::MAX { 0 } else { v }
    }

    /// Maximum step time in nanoseconds.
    #[must_use]
    pub fn max_ns(&self) -> u64 {
        self.max_ns.load(Ordering::Relaxed)
    }

    /// Last step time in nanoseconds.
    #[must_use]
    pub fn last_ns(&self) -> u64 {
        self.last_ns.load(Ordering::Relaxed)
    }

    /// Component name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Reset all statistics (does not change enabled state).
    pub fn reset_stats(&self) {
        self.steps.store(0, Ordering::Relaxed);
        self.total_ns.store(0, Ordering::Relaxed);
        self.min_ns.store(u64::MAX, Ordering::Relaxed);
        self.max_ns.store(0, Ordering::Relaxed);
        self.last_ns.store(0, Ordering::Relaxed);
    }

    /// Render statistics as a debug string.
    #[must_use]
    pub fn summary(&self) -> String {
        let steps = self.steps();
        if steps == 0 {
            return format!("{}: no steps recorded", self.name);
        }
        let mean = self.mean_ns();
        let min = self.min_ns();
        let max = self.max_ns();
        format!(
            "{}: {} steps, mean={:.2}µs, min={:.2}µs, max={:.2}µs",
            self.name,
            steps,
            mean as f64 / 1000.0,
            min as f64 / 1000.0,
            max as f64 / 1000.0,
        )
    }
}

impl std::fmt::Debug for StepTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepTimer")
            .field("name", &self.name)
            .field("enabled", &self.is_enabled())
            .field("steps", &self.steps())
            .field("total_ns", &self.total_ns())
            .field("min_ns", &self.min_ns())
            .field("max_ns", &self.max_ns())
            .field("last_ns", &self.last_ns())
            .finish()
    }
}

/// RAII guard that records the step duration when dropped.
pub struct StepGuard<'a> {
    timer: &'a StepTimer,
    start: Instant,
    step_id: Option<u64>,
}

impl Drop for StepGuard<'_> {
    fn drop(&mut self) {
        if self.step_id.is_some() {
            let duration = self.start.elapsed();
            self.timer.record(duration);
        }
    }
}

impl StepGuard<'_> {
    /// Elapsed time so far (without stopping).
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn disabled_timer_does_not_record() {
        let timer = StepTimer::new("test");
        assert!(!timer.is_enabled());

        {
            let _guard = timer.start(0);
            thread::sleep(Duration::from_micros(100));
        }

        assert_eq!(timer.steps(), 0);
        assert_eq!(timer.mean_ns(), 0);
    }

    #[test]
    fn enabled_timer_records_steps() {
        let timer = StepTimer::new("test");
        timer.set_enabled(true);

        {
            let _guard = timer.start(1);
            thread::sleep(Duration::from_micros(500));
        }
        {
            let _guard = timer.start(2);
            thread::sleep(Duration::from_micros(1000));
        }

        assert_eq!(timer.steps(), 2);
        assert!(timer.mean_ns() > 0);
        assert!(timer.min_ns() <= timer.max_ns());
    }

    #[test]
    fn reset_stats() {
        let timer = StepTimer::new("test");
        timer.set_enabled(true);

        {
            let _guard = timer.start(0);
        }
        assert!(timer.steps() > 0);

        timer.reset_stats();
        assert_eq!(timer.steps(), 0);
        assert_eq!(timer.mean_ns(), 0);
    }

    #[test]
    fn summary_empty() {
        let timer = StepTimer::new("test");
        assert_eq!(timer.summary(), "test: no steps recorded");
    }

    #[test]
    fn summary_with_data() {
        let timer = StepTimer::new("infe-kv");
        timer.set_enabled(true);
        {
            let _guard = timer.start(0);
        }
        let s = timer.summary();
        assert!(s.contains("infe-kv"));
        assert!(s.contains("1 steps"));
    }

    #[test]
    fn guard_elapsed() {
        let timer = StepTimer::new("test");
        timer.set_enabled(true);
        let guard = timer.start(0);
        thread::sleep(Duration::from_micros(200));
        let elapsed = guard.elapsed();
        assert!(elapsed.as_micros() >= 100);
    }
}
