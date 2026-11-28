//! The build identity this binary was compiled with.
//!
//! One source of truth shared by the CLI (`--version`) and the HTTP surface
//! (`/__bili/status`, `/__bili/plugin/runtime-info`, `/__bili/health`), so the
//! two can never disagree about what is running.
//!
//! The identity is intentionally stable: it is derived from the package version,
//! the checked-out commit and the cargo profile — never from a per-build
//! timestamp. Rebuilding the same commit must produce the same identity.
//!
//! It deliberately carries no paths, no environment values, no credentials and
//! no user data: `version` is a semver string from `Cargo.toml`, `commit` is a
//! short hex SHA, `profile` is a cargo profile name, `dirty` is a boolean.

use serde_json::{Value, json};

/// The package version, straight from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The commit recorded by `build.rs`. `unknown` when git was unavailable.
pub const COMMIT: &str = env!("BILI_BUILD_COMMIT");

/// The cargo profile recorded by `build.rs`, e.g. `release` or `debug`.
pub const PROFILE: &str = env!("BILI_BUILD_PROFILE");

/// Whether the working tree had uncommitted changes at build time.
pub const DIRTY: bool = matches!(env!("BILI_BUILD_DIRTY").as_bytes(), b"1");

/// A resolved build identity. Cloned freely; it is a handful of `&'static str`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildInfo {
    pub version: &'static str,
    pub commit: &'static str,
    pub profile: &'static str,
    pub dirty: bool,
}

impl BuildInfo {
    /// The identity of the currently running binary.
    pub fn current() -> Self {
        Self {
            version: VERSION,
            commit: COMMIT,
            profile: PROFILE,
            dirty: DIRTY,
        }
    }

    /// Render as a short human line: `0.1.0 (3fc289e589b9, release)`.
    ///
    /// A dirty build is marked so it can never be mistaken for a clean release.
    pub fn summary(&self) -> String {
        let dirty = if self.dirty { ", dirty" } else { "" };
        format!(
            "{} ({}, {}{})",
            self.version, self.commit, self.profile, dirty
        )
    }

    /// Render the multi-line CLI form used by `--version`.
    ///
    /// The first line stays `billion-context-proxy <version>`, which is what humans and
    /// `--version`-style greps have always looked for; the rest is additive.
    pub fn cli_text(&self) -> String {
        format!(
            "billion-context-proxy {}\ncommit {}\nprofile {}\ndirty {}",
            self.version, self.commit, self.profile, self.dirty
        )
    }

    /// The HTTP representation. Deliberately a fixed, small key set — see the
    /// module docs for what is intentionally absent.
    pub fn json(&self) -> Value {
        json!({
            "version": self.version,
            "commit": self.commit,
            "profile": self.profile,
            "dirty": self.dirty,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn looks_like_short_sha(value: &str) -> bool {
        (7..=40).contains(&value.len()) && value.bytes().all(|b| b.is_ascii_hexdigit())
    }

    #[test]
    fn version_comes_from_the_package_manifest() {
        assert_eq!(BuildInfo::current().version, env!("CARGO_PKG_VERSION"));
        assert!(
            VERSION.split('.').count() == 3,
            "version should be a three-part semver, got {VERSION:?}"
        );
        assert!(
            VERSION.split('.').all(|part| part.parse::<u32>().is_ok()),
            "every version segment should be numeric, got {VERSION:?}"
        );
    }

    #[test]
    fn commit_is_a_short_sha_or_the_unknown_placeholder() {
        assert!(!COMMIT.is_empty(), "commit must never be empty");
        assert_eq!(COMMIT.trim(), COMMIT, "commit must not carry padding");
        assert!(
            COMMIT == "unknown" || looks_like_short_sha(COMMIT),
            "commit should be hex or \"unknown\", got {COMMIT:?}"
        );
    }

    #[test]
    fn profile_is_a_plain_cargo_profile_name() {
        assert!(!PROFILE.is_empty(), "profile must never be empty");
        assert!(
            PROFILE
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
            "profile should be a bare cargo profile name, got {PROFILE:?}"
        );
    }

    #[test]
    fn status_json_carries_exactly_the_identity_fields_and_nothing_sensitive() {
        let value = BuildInfo::current().json();
        let object = value.as_object().expect("build identity is a JSON object");

        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["commit", "dirty", "profile", "version"]);

        assert!(object["version"].is_string());
        assert!(object["commit"].is_string());
        assert!(object["profile"].is_string());
        assert!(object["dirty"].is_boolean());

        // Nothing in the payload may look like a secret or a filesystem path.
        for (key, value) in object {
            let lowered = key.to_ascii_lowercase();
            for banned in [
                "key",
                "token",
                "secret",
                "password",
                "credential",
                "auth",
                "env",
            ] {
                assert!(
                    !lowered.contains(banned),
                    "build identity must not expose {banned:?} (found key {key:?})"
                );
            }
            if let Some(text) = value.as_str() {
                assert!(
                    !text.contains('\\') && !text.contains('/') && !text.contains(':'),
                    "build identity value {key:?} must not look like a path, got {text:?}"
                );
            }
        }
    }

    #[test]
    fn cli_text_and_status_json_report_the_same_identity() {
        let info = BuildInfo::current();
        let cli = info.cli_text();
        let lines: Vec<&str> = cli.lines().collect();
        assert_eq!(lines.len(), 4, "cli identity should be four lines: {cli:?}");
        assert_eq!(lines[0], format!("billion-context-proxy {}", info.version));
        assert_eq!(lines[1], format!("commit {}", info.commit));
        assert_eq!(lines[2], format!("profile {}", info.profile));
        assert_eq!(lines[3], format!("dirty {}", info.dirty));

        let json = info.json();
        assert_eq!(json["version"], info.version);
        assert_eq!(json["commit"], info.commit);
        assert_eq!(json["profile"], info.profile);
        assert_eq!(json["dirty"], info.dirty);
    }

    #[test]
    fn summary_is_compact_and_marks_dirty_builds() {
        let clean = BuildInfo {
            version: "9.9.9",
            commit: "0123456789ab",
            profile: "release",
            dirty: false,
        };
        assert_eq!(clean.summary(), "9.9.9 (0123456789ab, release)");

        let dirty = BuildInfo {
            dirty: true,
            ..clean
        };
        assert_eq!(dirty.summary(), "9.9.9 (0123456789ab, release, dirty)");
    }

    #[test]
    fn identity_does_not_change_between_reads() {
        // Guards against a timestamp or a counter sneaking back in: the identity
        // must be the same value however many times it is asked for.
        assert_eq!(BuildInfo::current(), BuildInfo::current());
        assert_eq!(BuildInfo::current().json(), BuildInfo::current().json());
    }
}
