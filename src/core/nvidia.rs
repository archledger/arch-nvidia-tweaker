// ═══════════════════════════════════════════════════════════════════════════════════════════
// Phase 24: NVIDIA driver-branch probe
// ═══════════════════════════════════════════════════════════════════════════════════════════
//
// We need exactly one fact about the installed NVIDIA driver: its major version. That
// number splits modern Arch NVIDIA behavior into two eras:
//
//   430 – 590: manual suspend/hibernate/resume systemd units are required; the
//              `NVreg_PreserveVideoMemoryAllocations=1` modprobe option is the flag
//              that tells the driver to keep framebuffer content across suspend.
//
//   595+:      kernel suspend notifiers (`NVreg_UseKernelSuspendNotifiers=1`) replaced
//              the old flag — it's still accepted but no longer honored. Arch's
//              `nvidia-utils` install hook ACTIVELY DISABLES the systemd units on
//              upgrade to 595+ ("upstream recommendation"), so any tool that enables
//              them will see its state silently reverted on every nvidia-utils bump.
//
// This module provides a pure parser + a runtime probe against `pacman -Q`. The query
// covers every per-branch package name Arch ships (nvidia-utils, nvidia-open-dkms,
// nvidia-580xx-utils, nvidia-580xx-dkms, nvidia-470xx-utils, nvidia-390xx-utils) so we
// don't depend on a specific driver flavor being installed — as long as SOME NVIDIA
// userspace is on disk, we get the branch.
//
// Consumers (power.rs) call `is_legacy_suspend_branch()` to decide whether to write
// the legacy `PreserveVideoMemoryAllocations` option and whether to enable the
// suspend/hibernate/resume units. Hosts with no NVIDIA installed return None →
// power actions are gated by `has_nvidia` in `run_actions` anyway, so None never
// reaches the legacy/modern decision.
// ═══════════════════════════════════════════════════════════════════════════════════════════

use std::process::Command;

/// Split `<major>.<minor>[.<patch>]` into a (major, minor) tuple. We don't track patch
/// or pkgrel — nothing downstream cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvidiaDriverVersion {
    pub major: u32,
    pub minor: u32,
}

impl NvidiaDriverVersion {
    /// True for 430 through 590 (the "old" era). 595+ is handled by kernel suspend
    /// notifiers and Arch's install hook disables the three nvidia-*.service units.
    pub fn is_legacy_suspend_branch(&self) -> bool {
        self.major < 595
    }

    /// Human-readable branch tag for log lines and diagnostic findings.
    pub fn branch_tag(&self) -> &'static str {
        match self.major {
            0..=429 => "unknown-legacy",
            430..=594 => "legacy-430-590",
            595..=u32::MAX => "modern-595+",
        }
    }
}

/// Pure: parse a package version string like `595.58.03-2`, `580.142-2`,
/// `470.256.02-1`, `390.157-13`. Accepts anything whose pre-dash portion is
/// `MAJOR.MINOR[.<anything>]`. Returns None for malformed inputs.
pub fn parse_driver_version(raw: &str) -> Option<NvidiaDriverVersion> {
    // Strip the pacman pkgrel suffix (the `-2` in `595.58.03-2`). Some AUR pins may
    // have no pkgrel — that's fine, the outer raw is unchanged.
    let dot_portion = raw.split('-').next()?;
    let mut parts = dot_portion.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts
        .next()
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);
    Some(NvidiaDriverVersion { major, minor })
}

/// Every NVIDIA userspace / kernel-module package name across every Arch + AUR
/// branch. Order doesn't matter — any hit yields the same branch classification.
const NVIDIA_PACKAGE_NAMES: &[&str] = &[
    // Official repo (modern open-modules path)
    "nvidia-utils",
    "lib32-nvidia-utils",
    "nvidia-open-dkms",
    "nvidia-open",
    // Formerly-official proprietary — still aliased via AUR / fallback systems
    "nvidia-dkms",
    // AUR legacy branches (Maxwell/Volta/Pascal on the 580 tree, Kepler on 470,
    // Fermi on 390).
    "nvidia-580xx-dkms",
    "nvidia-580xx-utils",
    "lib32-nvidia-580xx-utils",
    "nvidia-470xx-dkms",
    "nvidia-470xx-utils",
    "lib32-nvidia-470xx-utils",
    "nvidia-390xx-dkms",
    "nvidia-390xx-utils",
    "lib32-nvidia-390xx-utils",
];

/// Pure: given the raw output of `pacman -Q`, locate any NVIDIA driver package and
/// extract its version. Multiple matches are possible (e.g. nvidia-utils +
/// lib32-nvidia-utils) — their versions agree in practice, so we return the first.
pub fn detect_driver_version_from_pacman(pacman_q: &str) -> Option<NvidiaDriverVersion> {
    for line in pacman_q.lines() {
        let mut parts = line.split_whitespace();
        let Some(name) = parts.next() else { continue };
        let Some(version) = parts.next() else { continue };
        if NVIDIA_PACKAGE_NAMES.contains(&name) {
            if let Some(v) = parse_driver_version(version) {
                return Some(v);
            }
        }
    }
    None
}

/// Runtime: shell out to `pacman -Q`, delegate to the pure parser. Returns None on
/// any failure — downstream code treats "unknown driver version" as "don't apply
/// the 430-590-era quirks" (conservative: match Arch's modern install-hook defaults).
pub fn detect_driver_version_live() -> Option<NvidiaDriverVersion> {
    let out = Command::new("pacman").arg("-Q").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let body = String::from_utf8_lossy(&out.stdout);
    detect_driver_version_from_pacman(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_595_with_patch_and_pkgrel() {
        let v = parse_driver_version("595.58.03-2").unwrap();
        assert_eq!(v.major, 595);
        assert_eq!(v.minor, 58);
        assert!(!v.is_legacy_suspend_branch());
        assert_eq!(v.branch_tag(), "modern-595+");
    }

    #[test]
    fn parses_580_legacy_branch() {
        let v = parse_driver_version("580.142-2").unwrap();
        assert_eq!(v.major, 580);
        assert_eq!(v.minor, 142);
        assert!(v.is_legacy_suspend_branch(),
            "580 is in the 430-590 era — needs manual suspend units");
        assert_eq!(v.branch_tag(), "legacy-430-590");
    }

    #[test]
    fn parses_470xx_legacy_kepler_maxwell() {
        let v = parse_driver_version("470.256.02-1").unwrap();
        assert_eq!(v.major, 470);
        assert_eq!(v.minor, 256);
        assert!(v.is_legacy_suspend_branch());
    }

    #[test]
    fn parses_390xx_legacy_fermi() {
        let v = parse_driver_version("390.157-13").unwrap();
        assert_eq!(v.major, 390);
        assert!(v.is_legacy_suspend_branch());
    }

    #[test]
    fn parses_565_legacy_boundary() {
        // Right at the 430-590 ↔ 595+ boundary — 594 still uses the old flags.
        let v = parse_driver_version("594.0.0-1").unwrap();
        assert!(v.is_legacy_suspend_branch());
    }

    #[test]
    fn parses_future_600_as_modern() {
        let v = parse_driver_version("600.10.00-1").unwrap();
        assert!(!v.is_legacy_suspend_branch());
        assert_eq!(v.branch_tag(), "modern-595+");
    }

    #[test]
    fn parse_rejects_malformed_input() {
        assert!(parse_driver_version("").is_none());
        assert!(parse_driver_version("not-a-version").is_none());
        assert!(parse_driver_version("v595").is_none());
    }

    #[test]
    fn parse_tolerates_missing_minor() {
        // Hypothetical clean major-only version — treat minor as 0 rather than
        // rejecting, so the branch classification still works.
        let v = parse_driver_version("595").unwrap();
        assert_eq!(v.major, 595);
        assert_eq!(v.minor, 0);
    }

    // ── detect_driver_version_from_pacman ───────────────────────────────────────

    const PACMAN_Q_MODERN_OPEN: &str = "\
base 3-2
base-devel 1-2
nvidia-open-dkms 595.58.03-2
nvidia-utils 595.58.03-2
lib32-nvidia-utils 595.58.03-2
vulkan-icd-loader 1.4.319-1
";

    const PACMAN_Q_LEGACY_580: &str = "\
base 3-2
nvidia-580xx-dkms 580.142-2
nvidia-580xx-utils 580.142-2
lib32-nvidia-580xx-utils 580.142-2
";

    const PACMAN_Q_LEGACY_470: &str = "\
base 3-2
nvidia-470xx-dkms 470.256.02-1
nvidia-470xx-utils 470.256.02-1
";

    const PACMAN_Q_NO_NVIDIA: &str = "\
base 3-2
vulkan-radeon 26.0.5-1
mesa 26.0.5-1
";

    #[test]
    fn detects_modern_open_from_pacman() {
        let v = detect_driver_version_from_pacman(PACMAN_Q_MODERN_OPEN).unwrap();
        assert_eq!(v.major, 595);
        assert!(!v.is_legacy_suspend_branch());
    }

    #[test]
    fn detects_580_legacy_branch_from_pacman() {
        let v = detect_driver_version_from_pacman(PACMAN_Q_LEGACY_580).unwrap();
        assert_eq!(v.major, 580);
        assert!(v.is_legacy_suspend_branch());
    }

    #[test]
    fn detects_470_legacy_branch_from_pacman() {
        let v = detect_driver_version_from_pacman(PACMAN_Q_LEGACY_470).unwrap();
        assert_eq!(v.major, 470);
        assert!(v.is_legacy_suspend_branch());
    }

    #[test]
    fn returns_none_when_no_nvidia_installed() {
        assert!(detect_driver_version_from_pacman(PACMAN_Q_NO_NVIDIA).is_none());
    }

    #[test]
    fn returns_none_on_empty_pacman_output() {
        assert!(detect_driver_version_from_pacman("").is_none());
    }
}
