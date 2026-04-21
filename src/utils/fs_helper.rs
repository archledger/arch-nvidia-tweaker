use anyhow::{Context, Result};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Report returned by write operations so callers (CLI/GUI) can surface what happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeReport {
    /// Target already matches the desired state — no write performed.
    AlreadyApplied { detail: String },
    /// Target was created or updated. `backup` is `Some` when an existing file was replaced.
    Applied {
        detail: String,
        backup: Option<PathBuf>,
    },
    /// Dry-run: would have applied. No files written.
    Planned { detail: String },
}

impl std::fmt::Display for ChangeReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyApplied { detail } => write!(f, "[ok]      {detail}"),
            Self::Applied {
                detail,
                backup: None,
            } => write!(f, "[applied] {detail}"),
            Self::Applied {
                detail,
                backup: Some(bk),
            } => write!(f, "[applied] {detail} (backup: {})", bk.display()),
            Self::Planned { detail } => write!(f, "[planned] {detail}"),
        }
    }
}

/// Atomic write: write to a sibling temp file, fsync, then rename over the target.
/// Creates parent directory if missing.
///
/// **Security hardening (Phase 31 audit H1+H2):**
///   - Temp filename includes PID + nanosecond clock for unpredictability — a
///     non-root attacker cannot pre-plant a symlink at the temp path even with
///     directory listing access.
///   - Temp file is opened with `O_NOFOLLOW | O_EXCL` so a pre-planted symlink
///     causes the open to fail rather than silently follow.
///   - After rename, the parent directory is fsynced so the rename is durable
///     across crashes — critical for `/etc/kernel/cmdline` written immediately
///     before `mkinitcpio -P` rebuilds the UKI.
pub fn atomic_write<P: AsRef<Path>>(target: P, contents: &str) -> Result<()> {
    let target = target.as_ref();
    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating parent dir {}", parent.display()))?;
        }
    }

    // Phase 31 audit H1: use PID + ns-precision timestamp for the suffix so the
    // temp path is not predictable to a process that knows our PID.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let tmp = {
        let mut s = target.as_os_str().to_owned();
        s.push(format!(".tmp.{}.{}", std::process::id(), nanos));
        PathBuf::from(s)
    };

    {
        // Phase 31 audit H1: O_NOFOLLOW refuses to follow a pre-planted symlink
        // at the temp path; O_EXCL refuses to open if the target exists. Together
        // they make the temp-file create-and-write step TOCTOU-safe even with a
        // permissive parent dir.
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&tmp)
            .with_context(|| format!("creating {} (O_NOFOLLOW|O_EXCL)", tmp.display()))?;
        f.write_all(contents.as_bytes())
            .with_context(|| format!("writing {}", tmp.display()))?;
        f.sync_all()
            .with_context(|| format!("fsyncing {}", tmp.display()))?;
    }

    std::fs::rename(&tmp, target)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), target.display()))?;

    // Phase 31 audit H2: fsync the parent directory so the rename's directory-
    // entry change is durable. Without this, a crash between rename() and the
    // next sync could lose the rename and leave the dir entry pointing at the
    // old inode (or no inode). This matters most for /etc/kernel/cmdline: a
    // crash mid-mkinitcpio could otherwise produce an initramfs built from the
    // new cmdline but a /etc/kernel/cmdline that snaps back to the old content.
    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() {
            // Best-effort; not all filesystems support directory fsync. Warn on
            // failure but don't fail the write — the data file IS already
            // fsynced, which is the most-load-bearing guarantee.
            if let Ok(dir_handle) = std::fs::File::open(parent) {
                let _ = dir_handle.sync_all();
            }
        }
    }
    Ok(())
}

/// Copy `target` into `backup_dir` as `<basename>.bak-<unix-ts>`. Returns the backup path,
/// or `Ok(None)` if the source does not exist.
pub fn backup_to_dir<P: AsRef<Path>, Q: AsRef<Path>>(
    target: P,
    backup_dir: Q,
) -> Result<Option<PathBuf>> {
    let target = target.as_ref();
    if !target.exists() {
        return Ok(None);
    }
    let backup_dir = backup_dir.as_ref();
    std::fs::create_dir_all(backup_dir)
        .with_context(|| format!("creating backup dir {}", backup_dir.display()))?;

    let basename = target
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("no filename in {}", target.display()))?;

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let backup_path = backup_dir.join(format!("{}.bak-{}", basename.to_string_lossy(), ts));

    std::fs::copy(target, &backup_path)
        .with_context(|| format!("copying {} -> {}", target.display(), backup_path.display()))?;
    Ok(Some(backup_path))
}

/// Idempotent drop-in writer. If `target` already matches `contents` byte-for-byte, returns
/// `AlreadyApplied`. Otherwise, backs up an existing file (if any) and atomically replaces it.
/// When `dry_run` is true, no files are written and `Planned` is returned.
pub fn write_dropin<P: AsRef<Path>, Q: AsRef<Path>>(
    target: P,
    contents: &str,
    backup_dir: Q,
    dry_run: bool,
) -> Result<ChangeReport> {
    let target = target.as_ref();
    let existing = if target.exists() {
        Some(
            std::fs::read_to_string(target)
                .with_context(|| format!("reading {}", target.display()))?,
        )
    } else {
        None
    };

    if existing.as_deref() == Some(contents) {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!("{} already up to date", target.display()),
        });
    }

    let detail = if existing.is_some() {
        format!("update {}", target.display())
    } else {
        format!("create {}", target.display())
    };

    if dry_run {
        return Ok(ChangeReport::Planned { detail });
    }

    let backup = if existing.is_some() {
        backup_to_dir(target, backup_dir)?
    } else {
        None
    };
    atomic_write(target, contents)?;
    Ok(ChangeReport::Applied { detail, backup })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn atomic_write_creates_file_and_parent() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("nested/sub/out.txt");
        atomic_write(&target, "hello\n").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "hello\n");
    }

    #[test]
    fn atomic_write_replaces_existing() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("out.txt");
        std::fs::write(&target, "old").unwrap();
        atomic_write(&target, "new").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
    }

    #[test]
    fn backup_returns_none_for_missing_source() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("missing");
        let backup_dir = dir.path().join("bak");
        assert_eq!(backup_to_dir(&missing, &backup_dir).unwrap(), None);
    }

    #[test]
    fn backup_copies_existing_file() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("pacman.conf");
        std::fs::write(&src, "original").unwrap();
        let backup_dir = dir.path().join("bak");
        let backup = backup_to_dir(&src, &backup_dir).unwrap().unwrap();
        assert!(backup.exists());
        assert_eq!(std::fs::read_to_string(&backup).unwrap(), "original");
        assert!(backup
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("pacman.conf.bak-"));
    }

    #[test]
    fn write_dropin_creates_new_file() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("etc/x.conf");
        let bak = dir.path().join("bak");
        let report = write_dropin(&target, "hello\n", &bak, false).unwrap();
        match report {
            ChangeReport::Applied { backup, .. } => assert!(backup.is_none()),
            other => panic!("expected Applied, got {other:?}"),
        }
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "hello\n");
    }

    #[test]
    fn write_dropin_is_idempotent() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("x.conf");
        let bak = dir.path().join("bak");
        write_dropin(&target, "hello\n", &bak, false).unwrap();
        let report = write_dropin(&target, "hello\n", &bak, false).unwrap();
        assert!(matches!(report, ChangeReport::AlreadyApplied { .. }));
    }

    #[test]
    fn write_dropin_backs_up_before_replacing() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("x.conf");
        let bak = dir.path().join("bak");
        std::fs::write(&target, "v1").unwrap();
        let report = write_dropin(&target, "v2", &bak, false).unwrap();
        let backup_path = match report {
            ChangeReport::Applied { backup, .. } => backup.expect("backup path"),
            other => panic!("expected Applied, got {other:?}"),
        };
        assert_eq!(std::fs::read_to_string(&backup_path).unwrap(), "v1");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "v2");
    }

    #[test]
    fn write_dropin_dry_run_does_not_write() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("x.conf");
        let bak = dir.path().join("bak");
        let report = write_dropin(&target, "hello\n", &bak, true).unwrap();
        assert!(matches!(report, ChangeReport::Planned { .. }));
        assert!(!target.exists());
        assert!(!bak.exists());
    }
}
