//! Snapshot persistence.
//!
//! Two rules, both learned the hard way upstream:
//!
//! 1. **The snapshot carries a version.** A state file outlives the binary that
//!    wrote it; refusing to guess is cheaper than corrupting a conversation.
//! 2. **Writes are atomic.** Write a sibling temp file, fsync, rename. A process
//!    killed mid-write leaves the previous snapshot intact rather than a
//!    half-written one that fails to parse.

use std::fmt;

pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug)]
pub enum PersistError {
    Io(String),
    /// A snapshot from a newer binary. Refusing is the only safe answer.
    Version { found: u32, supported: u32 },
    Corrupt(String),
}

impl fmt::Display for PersistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PersistError::Io(e) => write!(f, "snapshot I/O failed: {e}"),
            PersistError::Version { found, supported } => write!(
                f,
                "snapshot version {found} is newer than this build supports ({supported})"
            ),
            PersistError::Corrupt(e) => write!(f, "snapshot is corrupt: {e}"),
        }
    }
}

impl std::error::Error for PersistError {}

#[cfg(feature = "serde")]
mod inner {
    use super::*;
    use crate::state::State;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    #[derive(serde::Serialize, serde::Deserialize)]
    struct Envelope {
        version: u32,
        state: State,
    }

    pub fn to_json(state: &State) -> Result<String, PersistError> {
        serde_json::to_string(&Envelope {
            version: FORMAT_VERSION,
            state: state.clone(),
        })
        .map_err(|e| PersistError::Corrupt(e.to_string()))
    }

    pub fn from_json(text: &str) -> Result<State, PersistError> {
        // Read the version before the body: a v2 file must be rejected even if
        // the body would happen to deserialize.
        #[derive(serde::Deserialize)]
        struct VersionOnly {
            version: u32,
        }
        let probe: VersionOnly =
            serde_json::from_str(text).map_err(|e| PersistError::Corrupt(e.to_string()))?;
        if probe.version > FORMAT_VERSION {
            return Err(PersistError::Version {
                found: probe.version,
                supported: FORMAT_VERSION,
            });
        }
        let envelope: Envelope =
            serde_json::from_str(text).map_err(|e| PersistError::Corrupt(e.to_string()))?;
        Ok(envelope.state)
    }

    pub fn save(state: &State, path: &Path) -> Result<(), PersistError> {
        let json = to_json(state)?;
        let temp = temp_sibling(path);
        {
            let mut file = std::fs::File::create(&temp).map_err(|e| PersistError::Io(e.to_string()))?;
            file.write_all(json.as_bytes())
                .map_err(|e| PersistError::Io(e.to_string()))?;
            file.sync_all().map_err(|e| PersistError::Io(e.to_string()))?;
        }
        std::fs::rename(&temp, path).map_err(|e| {
            let _ = std::fs::remove_file(&temp);
            PersistError::Io(e.to_string())
        })
    }

    pub fn load(path: &Path) -> Result<State, PersistError> {
        let text = std::fs::read_to_string(path).map_err(|e| PersistError::Io(e.to_string()))?;
        from_json(&text)
    }

    fn temp_sibling(path: &Path) -> PathBuf {
        let mut name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "state".into());
        name.push_str(".tmp");
        path.with_file_name(name)
    }
}

#[cfg(feature = "serde")]
pub use inner::{from_json, load, save, to_json};

#[cfg(test)]
#[cfg(feature = "serde")]
mod tests {
    use super::*;
    use crate::ids::{Boundary, MsgRef};
    use crate::state::{Block, Generation, State, Tier};

    fn sample() -> State {
        let mut state = State::new();
        state.refs.allocate(&crate::ids::MessageId::new("m1"));
        state.token_snapshot.insert(MsgRef::new(1).unwrap(), 42);
        state.blocks.push(Block {
            id: crate::ids::BlockId::new(1).unwrap(),
            run: 1,
            tier: Tier::T1,
            topic: Some("intro".into()),
            summary: "The user introduced themselves.".into(),
            direct: vec![crate::ids::MessageId::new("m1")],
            effective: vec![crate::ids::MessageId::new("m1")],
            direct_blocks: Vec::new(),
            compressed_tokens: 42,
            created_at: 7,
            survived: 2,
            generation: Generation::Young,
            active: true,
            expanded: false,
            restored_inline: false,
            span: Some((
                Boundary::Message(MsgRef::new(1).unwrap()),
                Boundary::Message(MsgRef::new(1).unwrap()),
            )),
        });
        state
    }

    #[test]
    fn round_trips_through_json() {
        let state = sample();
        let json = to_json(&state).unwrap();
        let back = from_json(&json).unwrap();
        assert_eq!(back.blocks.len(), 1);
        assert_eq!(back.blocks[0].summary, state.blocks[0].summary);
        // The wire form is what a human would debug, not an integer dump.
        assert!(json.contains("m00001"));
        assert!(json.contains("b1"));
    }

    #[test]
    fn newer_version_is_refused() {
        let json = to_json(&sample()).unwrap().replace(
            "\"version\":1",
            "\"version\":99",
        );
        match from_json(&json) {
            Err(PersistError::Version { found, supported }) => {
                assert_eq!(found, 99);
                assert_eq!(supported, FORMAT_VERSION);
            }
            other => panic!("expected a version refusal, got {other:?}"),
        }
    }

    #[test]
    fn atomic_save_leaves_no_temp_file() {
        let dir = std::env::temp_dir().join("billion-context-persist-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");
        save(&sample(), &path).unwrap();
        assert!(path.exists());
        assert!(!dir.join("state.json.tmp").exists());
        let back = load(&path).unwrap();
        assert_eq!(back.blocks.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
