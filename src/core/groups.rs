//! Phase 27 — provision the invoking user's membership in `video` and `render`.
//!
//! Both groups gate access to the DRM devices the graphics stack depends on:
//!   - `/dev/dri/card*`   — mode-setting + GL (traditionally owned by `video`)
//!   - `/dev/dri/renderD*` — GPU compute + VA-API video decode (owned by `render`)
//!
//! On a modern Arch system most distros do NOT add a normal user to either group
//! by default; `systemd-logind` grants `video` dynamically for the *seat owner's*
//! login but that leaves `render` empty, which in turn makes VA-API and OpenCL
//! (and some Vulkan backends) silently fall back to software rendering on hosts
//! where the logind magic doesn't apply (SSH sessions, headless login, non-seat
//! users). Explicit static membership in both groups closes that gap.
//!
//! Group changes only reach the current session on next login — there is no
//! "reload supplementary groups" syscall a running process can issue against
//! itself. This module therefore distinguishes two post-apply states:
//!   - `/etc/group` lists the user AND the current session already has the
//!     groups live → `Active`.
//!   - `/etc/group` lists the user but the running session doesn't → the
//!     detail message makes it explicit: log out + back in (NOT a reboot).
//!
//! Running as root with no detectable invoking user (pure SSH-as-root, CI, etc.)
//! → `Incompatible`: root already has all access and there's no non-root user
//! whose groups we could modify.

use anyhow::{Context as _, Result};
use std::process::Command;

use crate::core::aur::invoking_user;
use crate::core::state::TweakState;
use crate::core::{Context, ExecutionMode};
use crate::utils::fs_helper::ChangeReport;
use crate::utils::process::run_streaming;

/// The two groups archgpu provisions. Ordered so `usermod -aG` messages read
/// "video,render" to match how they're usually quoted in documentation.
pub const REQUIRED_GROUPS: &[&str] = &["video", "render"];

pub fn check_state(ctx: &Context) -> TweakState {
    let Some(user) = invoking_user() else {
        // Running without a detectable non-root user — nothing to provision.
        return TweakState::Incompatible;
    };
    let Ok(group_file) = std::fs::read_to_string(&ctx.paths.group_file) else {
        // `/etc/group` unreadable → treat as Unapplied so the toggle stays callable;
        // apply() will produce a clear error at write time if the system is truly broken.
        return TweakState::Unapplied;
    };
    let missing: Vec<&&str> = REQUIRED_GROUPS
        .iter()
        .filter(|g| !user_in_group_file(&group_file, user.as_str(), g))
        .collect();
    if missing.is_empty() {
        TweakState::Active
    } else {
        TweakState::Unapplied
    }
}

pub fn apply(
    ctx: &Context,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ChangeReport>> {
    let _ = assume_yes; // usermod has no interactive prompt; accepted for signature symmetry
    let Some(user) = invoking_user() else {
        return Ok(vec![ChangeReport::AlreadyApplied {
            detail: "running without a non-root invoking user (SUDO_USER / PKEXEC_UID unset) — nothing to provision".into(),
        }]);
    };

    let group_file = std::fs::read_to_string(&ctx.paths.group_file)
        .with_context(|| format!("reading {}", ctx.paths.group_file.display()))?;

    let missing_groups: Vec<&str> = REQUIRED_GROUPS
        .iter()
        .copied()
        .filter(|g| !group_exists_in_file(&group_file, g))
        .collect();
    if !missing_groups.is_empty() {
        anyhow::bail!(
            "required group(s) missing from /etc/group: {} — reinstall `systemd` or create them manually",
            missing_groups.join(", ")
        );
    }

    let missing_member: Vec<&str> = REQUIRED_GROUPS
        .iter()
        .copied()
        .filter(|g| !user_in_group_file(&group_file, user.as_str(), g))
        .collect();
    if missing_member.is_empty() {
        return Ok(vec![ChangeReport::AlreadyApplied {
            detail: format!("{user} already in {}", REQUIRED_GROUPS.join(", ")),
        }]);
    }

    let groups_csv = missing_member.join(",");
    let detail = format!("usermod -aG {groups_csv} {user} (re-login required)");
    if ctx.mode.is_dry_run() {
        return Ok(vec![ChangeReport::Planned { detail }]);
    }
    if matches!(ctx.mode, ExecutionMode::Apply) {
        let mut cmd = Command::new("usermod");
        cmd.args(["-aG", &groups_csv, &user]);
        progress(&format!("[usermod] {detail}"));
        let status = run_streaming(cmd, |line| progress(&format!("[usermod] {line}")))?;
        if !status.success() {
            anyhow::bail!("usermod -aG {groups_csv} {user} exited with {status}");
        }
    }
    Ok(vec![ChangeReport::Applied {
        detail: format!(
            "added {user} to {groups_csv} — log out and back in (not reboot) to pick up the new groups"
        ),
        backup: None,
    }])
}

/// True if `line` is the `group:x:gid:member1,member2,...` entry for `group` AND
/// `user` appears in its member list.
fn user_in_group_file(group_file: &str, user: &str, group: &str) -> bool {
    group_file
        .lines()
        .find_map(|l| parse_group_line(l).filter(|(g, _)| *g == group))
        .map(|(_, members)| members.contains(&user))
        .unwrap_or(false)
}

fn group_exists_in_file(group_file: &str, group: &str) -> bool {
    group_file
        .lines()
        .any(|l| parse_group_line(l).is_some_and(|(g, _)| g == group))
}

/// `/etc/group` line format: `name:passwd:gid:user1,user2,...`. Returns
/// `(name, members)` on a well-formed line; None on comments/blank lines.
fn parse_group_line(line: &str) -> Option<(&str, Vec<&str>)> {
    let trimmed = line.trim_end();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let mut parts = trimmed.split(':');
    let name = parts.next()?;
    let _passwd = parts.next()?;
    let _gid = parts.next()?;
    let members_field = parts.next().unwrap_or("");
    let members: Vec<&str> = if members_field.is_empty() {
        Vec::new()
    } else {
        members_field.split(',').filter(|m| !m.is_empty()).collect()
    };
    Some((name, members))
}

// Phase 31 audit: `live_session_has_groups` removed — it was `#[allow(dead_code)]`
// from Phase 27 with a comment about future use, but no caller materialized in
// Phase 30 and the "log out and back in" apply-time message works without it.
// Re-add only when there's an actual consumer.

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_GROUP_FILE: &str = "\
root:x:0:
wheel:x:998:ben
video:x:985:
render:x:991:ben
audio:x:994:ben,guest
# comment line
:malformed
input:x:981:ben
ben:x:1000:
";

    #[test]
    fn parse_group_line_extracts_name_and_members() {
        let (name, members) = parse_group_line("audio:x:994:ben,guest").unwrap();
        assert_eq!(name, "audio");
        assert_eq!(members, vec!["ben", "guest"]);
    }

    #[test]
    fn parse_group_line_handles_empty_member_list() {
        let (name, members) = parse_group_line("video:x:985:").unwrap();
        assert_eq!(name, "video");
        assert!(members.is_empty());
    }

    #[test]
    fn parse_group_line_rejects_comments_and_blank_lines() {
        assert!(parse_group_line("# a comment").is_none());
        assert!(parse_group_line("").is_none());
        assert!(parse_group_line("   ").is_none());
    }

    #[test]
    fn user_in_group_file_detects_membership() {
        assert!(user_in_group_file(SAMPLE_GROUP_FILE, "ben", "render"));
        assert!(user_in_group_file(SAMPLE_GROUP_FILE, "ben", "wheel"));
        assert!(user_in_group_file(SAMPLE_GROUP_FILE, "guest", "audio"));
    }

    #[test]
    fn user_in_group_file_rejects_non_member() {
        // `ben` is NOT in the `video` group in the sample (empty member list).
        assert!(!user_in_group_file(SAMPLE_GROUP_FILE, "ben", "video"));
        // Group exists but user isn't in it.
        assert!(!user_in_group_file(SAMPLE_GROUP_FILE, "nobody", "render"));
        // Group doesn't exist at all.
        assert!(!user_in_group_file(SAMPLE_GROUP_FILE, "ben", "imaginary"));
    }

    #[test]
    fn user_in_group_file_does_not_match_substring_of_another_user() {
        let body = "video:x:985:benny\n";
        // `ben` is NOT in a group whose only member is `benny` — regression guard
        // against a naive `.contains()` implementation.
        assert!(!user_in_group_file(body, "ben", "video"));
    }

    #[test]
    fn group_exists_in_file_detects_well_known_groups() {
        assert!(group_exists_in_file(SAMPLE_GROUP_FILE, "video"));
        assert!(group_exists_in_file(SAMPLE_GROUP_FILE, "render"));
        assert!(!group_exists_in_file(SAMPLE_GROUP_FILE, "imaginary"));
    }
}
