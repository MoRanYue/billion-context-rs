//! One crate-wide lock for tests that mutate the process environment.
//!
//! `std::env::set_var` is process-global, so two test modules editing different
//! variables can still race: one module's restore can land between another
//! module's set and its assertion. Each module used to hold its own private
//! mutex, which serialized only its own tests and left the cross-module window
//! open — `summarizer::tests::credential_falls_back_to_the_harness_credential_file`
//! failed intermittently because `d1b_runner`'s tests were writing
//! `ONEAGENT_MODEL_API_KEY` while it read the credential file.
//!
//! Every test that touches the environment takes [`lock`] for the whole span in
//! which the variable must hold its value, and puts the variable back through
//! [`restore`] before releasing the guard. Nothing here is compiled outside
//! `cfg(test)`, so production credential resolution is untouched.

use std::ffi::OsString;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// The single lock every environment-mutating test must hold.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Take the shared environment lock.
///
/// A poisoned lock is recovered rather than propagated: a panicking test has
/// already failed, and turning every later test into a second failure would
/// hide the one real problem.
pub fn lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A variable's prior value, for [`restore`].
#[derive(Debug)]
pub struct Prior(Option<OsString>);

impl Prior {
    /// Take the captured value out, leaving an empty (absent) marker.
    ///
    /// `Drop` impls cannot move fields out, so they use this and then [`restore_taken`].
    pub fn take(&mut self) -> Option<OsString> {
        self.0.take()
    }
}

/// Capture a variable so it can be put back exactly as it was.
///
/// `None` means the variable was absent, which must restore as absent rather
/// than as an empty string — the two are different to `var_os`.
pub fn capture(name: &str) -> Prior {
    Prior(std::env::var_os(name))
}

/// Put a variable back to its captured state.
///
/// # Safety
/// Caller must hold [`lock`]; the environment is process-global.
pub unsafe fn restore(name: &str, prior: Prior) {
    unsafe { restore_taken(name, prior.0) }
}

/// Put a variable back to a value already taken out of a [`Prior`].
///
/// # Safety
/// Caller must hold [`lock`]; the environment is process-global.
pub unsafe fn restore_taken(name: &str, value: Option<OsString>) {
    unsafe {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
}

/// Set a variable, to be paired with a [`Prior`] captured beforehand.
///
/// # Safety
/// Caller must hold [`lock`]; the environment is process-global.
pub unsafe fn set(name: &str, value: impl AsRef<std::ffi::OsStr>) {
    unsafe { std::env::set_var(name, value) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROBE: &str = "BILI_TEST_ENV_PROBE";

    #[test]
    fn a_lock_is_reentrant_across_calls_within_one_thread() {
        // Two sequential acquisitions must both succeed: the guard is released
        // when dropped, so a module taking it twice in a row cannot deadlock.
        {
            let _guard = lock();
        }
        let _guard = lock();
    }

    #[test]
    fn restore_puts_back_an_absent_variable_as_absent() {
        let _guard = lock();
        let prior = capture(PROBE);
        unsafe { set(PROBE, "present") };
        assert_eq!(
            std::env::var_os(PROBE).as_deref(),
            Some(std::ffi::OsStr::new("present"))
        );
        unsafe { restore(PROBE, prior) };
        // Absent and empty are different states; restoring must not invent "".
        assert!(std::env::var_os(PROBE).is_none());
    }

    #[test]
    fn restore_is_exact_for_a_variable_that_already_had_a_value() {
        let _guard = lock();
        let prior = capture(PROBE);
        unsafe { set(PROBE, "original") };
        let before = capture(PROBE);
        unsafe { set(PROBE, "overwritten") };
        unsafe { restore(PROBE, before) };
        assert_eq!(
            std::env::var_os(PROBE).as_deref(),
            Some(std::ffi::OsStr::new("original"))
        );
        unsafe { restore(PROBE, prior) };
    }
}
