use anyhow::{Context as _, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::core::Context;
use crate::utils::fs_helper::ChangeReport;
use crate::utils::process::run_streaming;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AurHelper {
    Yay,
    Paru,
}

impl AurHelper {
    pub fn name(self) -> &'static str {
        match self {
            Self::Yay => "yay",
            Self::Paru => "paru",
        }
    }
}

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn detect_helper() -> Option<AurHelper> {
    if command_exists("yay") {
        return Some(AurHelper::Yay);
    }
    if command_exists("paru") {
        return Some(AurHelper::Paru);
    }
    None
}

/// The non-root user that invoked the tool, if determinable from the environment.
///
/// Defensive validation: names that contain shell/path metacharacters are rejected even if
/// `$SUDO_USER` is set to them. `cmd.arg()` doesn't shell-interpret, so this is belt-and-
/// suspenders — but the user string is also used to build `/tmp/...-<user>` paths, and a
/// name like `foo/../bar` would make those paths point somewhere unexpected.
pub fn invoking_user() -> Option<String> {
    if let Ok(u) = std::env::var("SUDO_USER") {
        if !u.is_empty() && u != "root" && is_valid_username(&u) {
            return Some(u);
        }
    }
    if let Ok(uid_str) = std::env::var("PKEXEC_UID") {
        if let Ok(uid) = uid_str.parse::<u32>() {
            if uid != 0 {
                if let Some(name) = username_for_uid(uid) {
                    if is_valid_username(&name) {
                        return Some(name);
                    }
                }
            }
        }
    }
    None
}

fn is_valid_username(user: &str) -> bool {
    // POSIX-compatible conservative allowlist: length 1..=32, ASCII alphanumerics plus
    // `_` and `-`. Rejects `/`, `..`, whitespace, null bytes, shell metacharacters.
    !user.is_empty()
        && user.len() <= 32
        && user
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn username_for_uid(uid: u32) -> Option<String> {
    let out = Command::new("getent")
        .arg("passwd")
        .arg(uid.to_string())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .next()?
        .split(':')
        .next()
        .map(|s| s.to_string())
}

fn uid_for_username(user: &str) -> Option<u32> {
    let out = Command::new("getent")
        .arg("passwd")
        .arg(user)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().next()?.split(':').nth(2)?.parse().ok()
}

/// Resolve the graphical askpass helper for the current desktop session.
pub fn detect_askpass() -> Option<PathBuf> {
    if let Ok(existing) = std::env::var("SUDO_ASKPASS") {
        let pb = PathBuf::from(&existing);
        if is_executable(&pb) {
            return Some(pb);
        }
    }

    let desktop = std::env::var("XDG_CURRENT_DESKTOP")
        .unwrap_or_default()
        .to_ascii_uppercase();

    let preferred: &[&str] = if desktop.contains("KDE") || desktop.contains("PLASMA") {
        &[
            "/usr/bin/ksshaskpass",
            "/usr/lib/ssh/x11-ssh-askpass",
            "/usr/bin/ssh-askpass",
            "/usr/lib/ssh/ssh-askpass",
        ]
    } else if desktop.contains("GNOME") || desktop.contains("UNITY") {
        &[
            "/usr/libexec/seahorse/ssh-askpass",
            "/usr/lib/seahorse/ssh-askpass",
            "/usr/bin/ssh-askpass",
        ]
    } else if desktop.contains("HYPRLAND")
        || desktop.contains("SWAY")
        || desktop.contains("WLROOTS")
        || desktop.contains("NIRI")
    {
        &[
            "/usr/bin/ksshaskpass",
            "/usr/bin/ssh-askpass",
            "/usr/lib/ssh/ssh-askpass",
        ]
    } else if desktop.contains("XFCE") || desktop.contains("MATE") || desktop.contains("LXQT") {
        &[
            "/usr/bin/lxqt-openssh-askpass",
            "/usr/bin/ksshaskpass",
            "/usr/bin/ssh-askpass",
            "/usr/lib/ssh/ssh-askpass",
        ]
    } else {
        &[
            "/usr/libexec/seahorse/ssh-askpass",
            "/usr/bin/ksshaskpass",
            "/usr/bin/ssh-askpass",
            "/usr/lib/ssh/ssh-askpass",
            "/usr/bin/lxqt-openssh-askpass",
        ]
    };

    preferred
        .iter()
        .map(PathBuf::from)
        .find(|p| is_executable(p))
}

fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let Ok(meta) = std::fs::metadata(p) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    meta.permissions().mode() & 0o111 != 0
}

/// Manual yay-bin bootstrap.
pub fn ensure_yay(ctx: &Context, progress: &mut dyn FnMut(&str)) -> Result<ChangeReport> {
    if let Some(h) = detect_helper() {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!("{} already installed", h.name()),
        });
    }

    let Some(user) = invoking_user() else {
        anyhow::bail!(
            "Cannot bootstrap yay: no non-root invoking user detected. \
             SUDO_USER/PKEXEC_UID are unset — `makepkg` refuses to run as root."
        );
    };

    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned {
            detail: format!(
                "bootstrap yay-bin (git clone + makepkg as user {user}, pacman -U as root)"
            ),
        });
    }

    progress(&format!("[aur] bootstrapping yay-bin as user `{user}`"));

    progress("[aur] pacman -S --needed git base-devel");
    let mut cmd = Command::new("pacman");
    cmd.args(["-S", "--needed", "--noconfirm", "git", "base-devel"]);
    let status = run_streaming(cmd, &mut *progress)?;
    if !status.success() {
        anyhow::bail!("pacman -S git base-devel exited with {status}");
    }

    let tmpdir = PathBuf::from(format!("/tmp/archgpu-yay-bootstrap-{user}"));
    let _ = std::fs::remove_dir_all(&tmpdir);
    let status = Command::new("install")
        .args(["-d", "-m", "0755", "-o", &user, "-g", &user])
        .arg(&tmpdir)
        .status()
        .context("install(1)")?;
    if !status.success() {
        anyhow::bail!("install -d {} exited with {status}", tmpdir.display());
    }

    progress("[aur] git clone yay-bin.git");
    let mut cmd = Command::new("sudo");
    cmd.args(["-u", &user, "-H", "git", "clone", "--depth=1"]);
    cmd.arg("https://aur.archlinux.org/yay-bin.git");
    cmd.arg(&tmpdir);
    let status = run_streaming(cmd, &mut *progress)?;
    if !status.success() {
        anyhow::bail!("git clone exited with {status}");
    }

    progress("[aur] makepkg --nocheck --noconfirm");
    let mut cmd = Command::new("sudo");
    cmd.args(["-u", &user, "-H", "makepkg", "--nocheck", "--noconfirm"]);
    cmd.current_dir(&tmpdir);
    let status = run_streaming(cmd, &mut *progress)?;
    if !status.success() {
        anyhow::bail!("makepkg exited with {status}");
    }

    let built = find_built_yay_package(&tmpdir).context("makepkg produced no package")?;
    progress(&format!("[aur] pacman -U --noconfirm {}", built.display()));
    let mut cmd = Command::new("pacman");
    cmd.args(["-U", "--noconfirm"]);
    cmd.arg(&built);
    let status = run_streaming(cmd, &mut *progress)?;
    if !status.success() {
        anyhow::bail!("pacman -U exited with {status}");
    }

    Ok(ChangeReport::Applied {
        detail: format!(
            "bootstrapped yay-bin (built by {user} in {})",
            tmpdir.display()
        ),
        backup: None,
    })
}

fn find_built_yay_package(dir: &Path) -> Option<PathBuf> {
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let p = entry.path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("yay-bin") {
            continue;
        }
        if name.ends_with(".sig") {
            continue;
        }
        if name.ends_with(".pkg.tar.zst") || name.ends_with(".pkg.tar.xz") {
            return Some(p);
        }
    }
    None
}

/// Install AUR packages via a helper (Phase 7: askpass routing for GUI contexts).
pub fn install_aur_packages(
    ctx: &Context,
    packages: &[&str],
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    if packages.is_empty() {
        return Ok(ChangeReport::AlreadyApplied {
            detail: "no AUR packages requested".into(),
        });
    }
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned {
            detail: format!("<helper> -S --needed {}", packages.join(" ")),
        });
    }

    if detect_helper().is_none() {
        let _ = ensure_yay(ctx, progress)?;
    }
    let helper = detect_helper()
        .context("AUR helper still unavailable after yay bootstrap — check the log above")?;
    let user = invoking_user().context(
        "cannot locate non-root invoking user to run the AUR helper (makepkg refuses as root)",
    )?;
    let uid = uid_for_username(&user).unwrap_or(1000);
    let askpass = detect_askpass();

    match askpass.as_ref() {
        Some(p) => progress(&format!(
            "[aur] using askpass for yay's inner sudo: {}",
            p.display()
        )),
        None => progress(
            "[aur] no graphical askpass found — yay's inner sudo may prompt on TTY; \
             consider installing ksshaskpass (KDE/Wayland) or seahorse (GNOME)",
        ),
    }
    progress(&format!(
        "[aur] sudo -u {user} {} -S --needed {}",
        helper.name(),
        packages.join(" ")
    ));

    let mut cmd = Command::new("sudo");
    cmd.args(["-u", &user, "-H"]);
    cmd.arg("env");
    cmd.arg(format!("XDG_RUNTIME_DIR=/run/user/{uid}"));
    if let Ok(v) = std::env::var("DISPLAY") {
        cmd.arg(format!("DISPLAY={v}"));
    }
    if let Ok(v) = std::env::var("WAYLAND_DISPLAY") {
        cmd.arg(format!("WAYLAND_DISPLAY={v}"));
    }
    if let Ok(v) = std::env::var("XAUTHORITY") {
        cmd.arg(format!("XAUTHORITY={v}"));
    }
    if let Ok(v) = std::env::var("XDG_CURRENT_DESKTOP") {
        cmd.arg(format!("XDG_CURRENT_DESKTOP={v}"));
    }
    if let Ok(v) = std::env::var("XDG_SESSION_TYPE") {
        cmd.arg(format!("XDG_SESSION_TYPE={v}"));
    }
    if let Some(p) = askpass.as_ref() {
        cmd.arg(format!("SUDO_ASKPASS={}", p.display()));
    }

    cmd.arg(helper.name());
    cmd.args(["-S", "--needed"]);
    if assume_yes {
        cmd.args([
            "--noconfirm",
            "--answerclean",
            "None",
            "--answerdiff",
            "None",
        ]);
    }
    if askpass.is_some() {
        cmd.args(["--sudoflags", "-A"]);
    }
    cmd.args(packages);

    let status = run_streaming(cmd, &mut *progress)?;
    if !status.success() {
        anyhow::bail!("{} exited with {status}", helper.name());
    }
    Ok(ChangeReport::Applied {
        detail: format!("installed from AUR: {}", packages.join(" ")),
        backup: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;

    #[test]
    fn is_executable_rejects_non_executable() {
        let dir = tempdir().unwrap();
        let plain = dir.path().join("plain.txt");
        std::fs::write(&plain, "hi").unwrap();
        assert!(!is_executable(&plain));
    }

    #[test]
    fn is_executable_accepts_mode_755() {
        let dir = tempdir().unwrap();
        let exe = dir.path().join("askpass");
        std::fs::write(&exe, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(is_executable(&exe));
    }

    #[test]
    fn is_executable_rejects_missing() {
        let dir = tempdir().unwrap();
        assert!(!is_executable(&dir.path().join("missing")));
    }
}
