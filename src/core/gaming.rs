use anyhow::{Context as _, Result};
use std::collections::HashSet;
use std::process::Command;

use crate::core::aur;
use crate::core::gpu::{GpuInventory, NvidiaGeneration, PackageSource};
use crate::core::state::TweakState;
use crate::core::{Context, ExecutionMode};
use crate::utils::fs_helper::{atomic_write, backup_to_dir, write_dropin, ChangeReport};
use crate::utils::process::run_streaming;

const SYSCTL_DROPIN_FILE: &str = "99-gaming.conf";
const SYSCTL_CONTENT: &str = "\
# Managed by archgpu — raises the mmap ceiling for modern games
# (Star Citizen, Hogwarts Legacy, Apex, etc.).
vm.max_map_count = 1048576
";

const ALWAYS_ON_GAMING_PACKAGES: &[&str] = &[
    "vulkan-icd-loader",
    "lib32-vulkan-icd-loader",
    "gamemode",
    "lib32-gamemode",
    "mangohud",
    "lib32-mangohud",
];

/// Gaming setup is Applied when: `[multilib]` is enabled AND every required repo package
/// for the detected GPU vendor(s) + the always-on set (vulkan-icd-loader / gamemode /
/// mangohud + their lib32 variants) is installed per `pacman -Qq`.
///
/// Never Incompatible — gaming setup is a universal improvement regardless of GPU vendor.
pub fn check_state(ctx: &Context, gpus: &GpuInventory) -> TweakState {
    if !is_multilib_enabled(&ctx.paths.pacman_conf) {
        return TweakState::Unapplied;
    }
    let expected = resolve_gaming_packages(gpus);
    let Some(installed) = pacman_query_installed_set(&expected) else {
        // pacman unavailable — can't verify; assume Unapplied so the button stays available.
        return TweakState::Unapplied;
    };
    if expected.iter().all(|p| installed.contains(p.as_str())) {
        // Purely file/package-level; no kernel probe applies. "Active" == "packages are
        // installed and multilib is on".
        TweakState::Active
    } else {
        TweakState::Unapplied
    }
}

fn pacman_query_installed_set(names: &[String]) -> Option<HashSet<String>> {
    let out = Command::new("pacman").arg("-Qq").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let expected: HashSet<&str> = names.iter().map(String::as_str).collect();
    Some(
        body.lines()
            .filter(|l| expected.contains(l.trim()))
            .map(|l| l.trim().to_string())
            .collect(),
    )
}

pub fn apply(
    ctx: &Context,
    gpus: &GpuInventory,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ChangeReport>> {
    let mut reports = Vec::new();
    reports.push(enable_multilib(ctx)?);
    reports.push(write_sysctl_dropin(ctx)?);
    reports.push(install_official_packages(ctx, gpus, assume_yes, progress)?);

    let aur_pkgs = resolve_aur_packages(gpus);
    if !aur_pkgs.is_empty() {
        let refs: Vec<&str> = aur_pkgs.iter().map(String::as_str).collect();
        reports.push(aur::install_aur_packages(ctx, &refs, assume_yes, progress)?);
    }

    Ok(reports)
}

fn write_sysctl_dropin(ctx: &Context) -> Result<ChangeReport> {
    let target = ctx.paths.sysctl_d.join(SYSCTL_DROPIN_FILE);
    write_dropin(
        &target,
        SYSCTL_CONTENT,
        &ctx.paths.backup_dir,
        ctx.mode.is_dry_run(),
    )
}

fn enable_multilib(ctx: &Context) -> Result<ChangeReport> {
    let path = &ctx.paths.pacman_conf;
    let original =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    let (new_text, changed) = uncomment_multilib(&original);
    if !changed {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!("{}: [multilib] already enabled", path.display()),
        });
    }
    let detail = format!("{}: uncomment [multilib]", path.display());
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned { detail });
    }
    let backup = backup_to_dir(path, &ctx.paths.backup_dir)?;
    atomic_write(path, &new_text)?;
    Ok(ChangeReport::Applied { detail, backup })
}

fn install_official_packages(
    ctx: &Context,
    gpus: &GpuInventory,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    let packages = resolve_gaming_packages(gpus);
    let detail = format!("pacman -Syu --needed {}", packages.join(" "));

    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned { detail });
    }
    if matches!(ctx.mode, ExecutionMode::Apply) {
        let mut cmd = Command::new("pacman");
        cmd.arg("-Syu").arg("--needed");
        if assume_yes {
            cmd.arg("--noconfirm");
        }
        cmd.args(&packages);
        progress(&format!("[pacman] {detail}"));
        let status = run_streaming(cmd, |line| progress(&format!("[pacman] {line}")))?;
        if !status.success() {
            anyhow::bail!("pacman exited with {status}");
        }
    }
    Ok(ChangeReport::Applied {
        detail,
        backup: None,
    })
}

/// Official (repo) packages to install for gaming.
pub fn resolve_gaming_packages(gpus: &GpuInventory) -> Vec<String> {
    let mut pkgs: Vec<String> = ALWAYS_ON_GAMING_PACKAGES
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    let mut add = |pkg: &str| {
        let s = pkg.to_string();
        if !pkgs.contains(&s) {
            pkgs.push(s);
        }
    };

    if let Some(nv) = gpus.primary_nvidia() {
        if let Some(rec) = nv.recommended_nvidia_package() {
            if rec.source == PackageSource::Official {
                add(rec.package);
            }
        }
        add("nvidia-utils");
        add("lib32-nvidia-utils");
        add("nvidia-settings");
        // Phase 15: VA-API via the NVIDIA-provided backend — required for modern
        // video acceleration in Firefox, mpv, etc. Replaces the old vdpau-va-gl
        // shim. Shipped as a separate package from nvidia-utils.
        add("libva-nvidia-driver");
        if gpus.is_hybrid() {
            add("nvidia-prime");
        }
    }
    if gpus.has_amd() {
        // RADV (vulkan-radeon) is Mesa's Vulkan driver — faster and more widely
        // tested in Proton than AMD's own AMDVLK. Sanitation code warns if the
        // user has amdvlk/lib32-amdvlk installed alongside.
        add("vulkan-radeon");
        add("lib32-vulkan-radeon");
        // Phase 15: VA-API via Mesa for AMD. Replaces the VDPAU path that Mesa
        // 25 removed — mesa-vdpau is sanitized-against below.
        add("libva-mesa-driver");
        add("lib32-libva-mesa-driver");
    }
    if gpus.has_intel() {
        add("vulkan-intel");
        add("lib32-vulkan-intel");
        // Phase 15: intel-media-driver (iHD) is the modern Gen8+ VA-API driver.
        // Gen4-7 hosts need libva-intel-driver (legacy i965) — anyone running this
        // tool on a Haswell-or-older Intel iGPU in 2026 can install that manually.
        add("intel-media-driver");
    }

    pkgs
}

// ── Phase 15 Sanitation: detect legacy/conflicting packages ─────────────────────────────────

/// A legacy/conflicting package that should be removed or replaced. Surfaced in the GUI as
/// a warning banner and in `--diagnose` as a Finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SanitationWarning {
    pub title: String,
    pub detail: String,
    pub remediation: String,
}

/// Walk the detected GPU inventory + a snapshot of installed packages, return any sanitation
/// concerns. Pure function, testable without `pacman`.
pub fn sanitation_warnings_from_installed(
    gpus: &GpuInventory,
    installed: &HashSet<String>,
) -> Vec<SanitationWarning> {
    let mut out = Vec::new();

    // The AMDVLK trap — AMDVLK is AMD's official Vulkan driver but in Proton/DXVK
    // benchmarks RADV (vulkan-radeon, ships with Mesa) is consistently faster and
    // more widely tested. Worse: if BOTH are installed, applications see two ICDs
    // and the wrong one may win via VK_ICD_FILENAMES ordering, causing hard-to-
    // diagnose perf regressions.
    if gpus.has_amd() {
        let names: Vec<&str> = ["amdvlk", "lib32-amdvlk"]
            .into_iter()
            .filter(|n| installed.contains(*n))
            .collect();
        if !names.is_empty() {
            out.push(SanitationWarning {
                title: "AMDVLK Vulkan driver installed alongside RADV".into(),
                detail: format!(
                    "Found: {}. AMD's official Vulkan driver (AMDVLK) is generally slower than Mesa's RADV \
                     for gaming under Proton/Steam, and having BOTH installed can let the wrong ICD win.",
                    names.join(", ")
                ),
                remediation: format!(
                    "sudo pacman -Rns {} (keep vulkan-radeon / lib32-vulkan-radeon — already installed by --apply-gaming)",
                    names.join(" ")
                ),
            });
        }
    }

    // The Intel DDX trap — xf86-video-intel has been deprecated upstream for years.
    // The generic Xorg `modesetting` driver (shipped in xorg-server) is the
    // recommended path for Gen4+ Intel iGPUs and is what Arch's wiki points to.
    // The DDX driver causes tearing + hangs on newer kernels, and has no Wayland story.
    if gpus.has_intel() && installed.contains("xf86-video-intel") {
        out.push(SanitationWarning {
            title: "Legacy xf86-video-intel DDX installed".into(),
            detail: "`xf86-video-intel` was deprecated upstream. The modesetting driver (shipped in \
                     xorg-server) is the modern replacement for Gen4+ Intel iGPUs — it supports KMS, \
                     Wayland, and avoids the tearing and hangs the DDX produces on recent kernels."
                .into(),
            remediation: "sudo pacman -Rns xf86-video-intel".into(),
        });
    }

    // Mesa 25 legacy — VDPAU for the Gallium drivers (AMD / Intel / Nouveau) was
    // removed in Mesa 25. The `mesa-vdpau` package still exists on Arch for
    // transition but no longer ships usable drivers; applications should use
    // VA-API (libva-mesa-driver, intel-media-driver, libva-nvidia-driver) instead.
    let vdpau_found: Vec<&str> = ["mesa-vdpau", "lib32-mesa-vdpau"]
        .into_iter()
        .filter(|n| installed.contains(*n))
        .collect();
    if !vdpau_found.is_empty() {
        out.push(SanitationWarning {
            title: "Legacy Mesa VDPAU packages installed".into(),
            detail: format!(
                "Found: {}. Mesa 25 removed in-tree VDPAU support for Gallium drivers. VA-API \
                 (libva-mesa-driver / intel-media-driver / libva-nvidia-driver) is the modern path.",
                vdpau_found.join(", ")
            ),
            remediation: format!("sudo pacman -Rns {}", vdpau_found.join(" ")),
        });
    }

    out
}

/// Runtime version that queries pacman for currently-installed packages, then delegates to
/// the pure `sanitation_warnings_from_installed` for the actual logic.
pub fn sanitation_warnings(gpus: &GpuInventory) -> Vec<SanitationWarning> {
    let candidates: &[&str] = &[
        "amdvlk",
        "lib32-amdvlk",
        "xf86-video-intel",
        "mesa-vdpau",
        "lib32-mesa-vdpau",
    ];
    let owned: Vec<String> = candidates.iter().map(|s| s.to_string()).collect();
    let installed = pacman_query_installed_set(&owned).unwrap_or_default();
    sanitation_warnings_from_installed(gpus, &installed)
}

/// AUR packages the tool will build+install via yay. Empty for Turing+ hosts.
pub fn resolve_aur_packages(gpus: &GpuInventory) -> Vec<String> {
    let Some(nv) = gpus.primary_nvidia() else {
        return Vec::new();
    };
    let gen = nv.nvidia_gen.unwrap_or(NvidiaGeneration::Unknown);

    match gen {
        NvidiaGeneration::Maxwell | NvidiaGeneration::Kepler => vec![
            "nvidia-470xx-dkms".into(),
            "nvidia-470xx-utils".into(),
            "lib32-nvidia-470xx-utils".into(),
        ],
        NvidiaGeneration::Fermi => vec![
            "nvidia-390xx-dkms".into(),
            "nvidia-390xx-utils".into(),
            "lib32-nvidia-390xx-utils".into(),
        ],
        _ => Vec::new(),
    }
}

/// Read-only check: is the `[multilib]` repository already enabled in `pacman_conf`?
pub fn is_multilib_enabled<P: AsRef<std::path::Path>>(pacman_conf: P) -> bool {
    let Ok(body) = std::fs::read_to_string(pacman_conf.as_ref()) else {
        return false;
    };
    let mut in_multilib = false;
    for line in body.lines() {
        let t = line.trim();
        if t == "[multilib]" {
            in_multilib = true;
            continue;
        }
        if in_multilib && t.starts_with('[') {
            in_multilib = false;
        }
        if in_multilib && t.starts_with("Include") {
            return true;
        }
    }
    false
}

pub fn uncomment_multilib(original: &str) -> (String, bool) {
    enum State {
        Scanning,
        InMultilib,
        Done,
    }

    let mut out: Vec<String> = Vec::with_capacity(original.lines().count() + 1);
    let mut state = State::Scanning;
    let mut changed = false;

    for line in original.lines() {
        let t = line.trim();
        match state {
            State::Scanning => {
                if t == "#[multilib]" {
                    out.push("[multilib]".to_string());
                    state = State::InMultilib;
                    changed = true;
                } else if t == "[multilib]" {
                    out.push(line.to_string());
                    state = State::InMultilib;
                } else {
                    out.push(line.to_string());
                }
            }
            State::InMultilib => {
                if is_commented_include(t) {
                    out.push(uncomment_include(t));
                    state = State::Done;
                    changed = true;
                } else if is_uncommented_include(t) || t.starts_with('[') {
                    // Either we've hit the Include line (already uncommented) or the next
                    // section header — in both cases we're done scanning multilib.
                    out.push(line.to_string());
                    state = State::Done;
                } else {
                    out.push(line.to_string());
                }
            }
            State::Done => out.push(line.to_string()),
        }
    }

    let mut result = out.join("\n");
    if original.ends_with('\n') {
        result.push('\n');
    }
    (result, changed)
}

fn is_commented_include(line: &str) -> bool {
    let Some(rest) = line.strip_prefix('#') else {
        return false;
    };
    let rest = rest.trim_start();
    rest.starts_with("Include") && rest.contains('=')
}

fn is_uncommented_include(line: &str) -> bool {
    line.starts_with("Include") && line.contains('=')
}

fn uncomment_include(line: &str) -> String {
    line.trim_start_matches('#').trim_start().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::{GpuInfo, GpuVendor, NvidiaGeneration};

    fn inv(gpus: Vec<GpuInfo>) -> GpuInventory {
        GpuInventory { gpus }
    }

    fn nvidia(gen: NvidiaGeneration, device_id: u16) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Nvidia,
            vendor_id: 0x10de,
            device_id,
            pci_address: "0000:01:00.0".into(),
            vendor_name: "NVIDIA".into(),
            product_name: format!("dev {device_id:04x}"),
            kernel_driver: None,
            is_integrated: false,
            nvidia_gen: Some(gen),
        }
    }

    fn intel(device_id: u16) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Intel,
            vendor_id: 0x8086,
            device_id,
            pci_address: "0000:00:02.0".into(),
            vendor_name: "Intel".into(),
            product_name: format!("iGPU {device_id:04x}"),
            kernel_driver: None,
            is_integrated: true,
            nvidia_gen: None,
        }
    }

    fn amd(device_id: u16) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Amd,
            vendor_id: 0x1002,
            device_id,
            pci_address: "0000:03:00.0".into(),
            vendor_name: "AMD".into(),
            product_name: format!("Radeon {device_id:04x}"),
            kernel_driver: None,
            is_integrated: false,
            nvidia_gen: None,
        }
    }

    #[test]
    fn intel_only_gets_no_nvidia_packages() {
        let pkgs = resolve_gaming_packages(&inv(vec![intel(0x64a0)]));
        assert!(!pkgs.iter().any(|p| p.starts_with("nvidia")));
        assert!(pkgs.contains(&"vulkan-intel".to_string()));
        // Phase 15: Intel VA-API via intel-media-driver.
        assert!(pkgs.contains(&"intel-media-driver".to_string()));
    }

    #[test]
    fn amd_gets_libva_mesa_driver() {
        let pkgs = resolve_gaming_packages(&inv(vec![amd(0x73bf)]));
        // Phase 15: AMD VA-API via Mesa.
        assert!(pkgs.contains(&"libva-mesa-driver".to_string()));
        assert!(pkgs.contains(&"lib32-libva-mesa-driver".to_string()));
        assert!(pkgs.contains(&"vulkan-radeon".to_string()));
        assert!(
            !pkgs.iter().any(|p| p == "amdvlk"),
            "never recommend AMDVLK"
        );
    }

    #[test]
    fn nvidia_gets_libva_nvidia_driver() {
        let pkgs = resolve_gaming_packages(&inv(vec![nvidia(NvidiaGeneration::Ada, 0x2684)]));
        // Phase 15: NVIDIA VA-API via libva-nvidia-driver.
        assert!(pkgs.contains(&"libva-nvidia-driver".to_string()));
    }

    #[test]
    fn nvidia_ada_desktop_gets_open_dkms_but_not_prime() {
        let pkgs = resolve_gaming_packages(&inv(vec![nvidia(NvidiaGeneration::Ada, 0x2684)]));
        assert!(pkgs.contains(&"nvidia-open-dkms".to_string()));
        assert!(!pkgs.iter().any(|p| p == "nvidia-prime"));
    }

    #[test]
    fn hybrid_gets_prime() {
        let pkgs = resolve_gaming_packages(&inv(vec![
            intel(0x3e9b),
            nvidia(NvidiaGeneration::Ampere, 0x25a2),
        ]));
        assert!(pkgs.contains(&"nvidia-prime".to_string()));
    }

    #[test]
    fn pascal_gets_legacy_nvidia_dkms_not_open() {
        let pkgs = resolve_gaming_packages(&inv(vec![nvidia(NvidiaGeneration::Pascal, 0x1B06)]));
        assert!(pkgs.contains(&"nvidia-dkms".to_string()));
        assert!(!pkgs.iter().any(|p| p == "nvidia-open-dkms"));
    }

    #[test]
    fn maxwell_official_list_excludes_aur_driver() {
        let pkgs = resolve_gaming_packages(&inv(vec![nvidia(NvidiaGeneration::Maxwell, 0x13C2)]));
        assert!(!pkgs.iter().any(|p| p.contains("470xx")));
    }

    #[test]
    fn maxwell_aur_list_contains_470xx_dkms_and_utils() {
        let aur = resolve_aur_packages(&inv(vec![nvidia(NvidiaGeneration::Maxwell, 0x13C2)]));
        assert!(aur.iter().any(|p| p == "nvidia-470xx-dkms"));
        assert!(aur.iter().any(|p| p == "nvidia-470xx-utils"));
        assert!(aur.iter().any(|p| p == "lib32-nvidia-470xx-utils"));
    }

    #[test]
    fn fermi_aur_list_is_390xx() {
        let aur = resolve_aur_packages(&inv(vec![nvidia(NvidiaGeneration::Fermi, 0x0DC4)]));
        assert!(aur.iter().any(|p| p == "nvidia-390xx-dkms"));
    }

    #[test]
    fn turing_or_newer_has_no_aur_requirements() {
        assert!(
            resolve_aur_packages(&inv(vec![nvidia(NvidiaGeneration::Turing, 0x1E04)])).is_empty()
        );
    }

    #[test]
    fn non_nvidia_never_requires_aur() {
        assert!(resolve_aur_packages(&inv(vec![intel(0x64a0)])).is_empty());
        assert!(resolve_aur_packages(&inv(vec![amd(0x73bf)])).is_empty());
    }

    const SAMPLE_COMMENTED: &str = "\
[core]
Include = /etc/pacman.d/mirrorlist

#[multilib]
#Include = /etc/pacman.d/mirrorlist
";

    const SAMPLE_ENABLED: &str = "\
[core]
Include = /etc/pacman.d/mirrorlist

[multilib]
Include = /etc/pacman.d/mirrorlist
";

    #[test]
    fn uncomments_multilib_section_and_include() {
        let (out, changed) = uncomment_multilib(SAMPLE_COMMENTED);
        assert!(changed);
        assert!(out.contains("\n[multilib]\n"));
    }

    #[test]
    fn idempotent_when_already_enabled() {
        let (_, changed) = uncomment_multilib(SAMPLE_ENABLED);
        assert!(!changed);
    }

    #[test]
    fn is_multilib_enabled_detects_active_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pacman.conf");
        std::fs::write(&path, SAMPLE_ENABLED).unwrap();
        assert!(is_multilib_enabled(&path));
    }

    // Phase 15: sanitation warnings ──────────────────────────────────────────────────────

    use std::collections::HashSet;

    fn installed_set(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn sanitation_no_warnings_on_clean_amd() {
        let w = sanitation_warnings_from_installed(&inv(vec![amd(0x73bf)]), &installed_set(&[]));
        assert!(w.is_empty());
    }

    #[test]
    fn sanitation_flags_amdvlk_on_amd_host() {
        let w = sanitation_warnings_from_installed(
            &inv(vec![amd(0x73bf)]),
            &installed_set(&["amdvlk", "lib32-amdvlk"]),
        );
        assert_eq!(w.len(), 1);
        assert!(w[0].title.contains("AMDVLK"));
        assert!(w[0].remediation.contains("amdvlk"));
    }

    #[test]
    fn sanitation_ignores_amdvlk_on_non_amd_host() {
        // If someone has amdvlk installed but no AMD GPU, that's weird but not our concern.
        let w = sanitation_warnings_from_installed(
            &inv(vec![intel(0x64a0)]),
            &installed_set(&["amdvlk"]),
        );
        assert!(w.iter().all(|x| !x.title.contains("AMDVLK")));
    }

    #[test]
    fn sanitation_flags_xf86_video_intel() {
        let w = sanitation_warnings_from_installed(
            &inv(vec![intel(0x64a0)]),
            &installed_set(&["xf86-video-intel"]),
        );
        assert!(w.iter().any(|x| x.title.contains("xf86-video-intel")));
    }

    #[test]
    fn sanitation_flags_mesa_vdpau_regardless_of_gpu() {
        // VDPAU is a cross-vendor legacy — flag it on any host that has it.
        let w = sanitation_warnings_from_installed(
            &inv(vec![nvidia(NvidiaGeneration::Ada, 0x2684)]),
            &installed_set(&["mesa-vdpau"]),
        );
        assert!(w.iter().any(|x| x.title.contains("VDPAU")));
    }

    #[test]
    fn sanitation_clean_system_no_warnings() {
        let w = sanitation_warnings_from_installed(
            &inv(vec![nvidia(NvidiaGeneration::Ada, 0x2684)]),
            &installed_set(&["vulkan-radeon", "gamemode", "libva-nvidia-driver"]),
        );
        assert!(w.is_empty());
    }

    #[test]
    fn is_multilib_enabled_detects_commented_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pacman.conf");
        std::fs::write(&path, SAMPLE_COMMENTED).unwrap();
        assert!(!is_multilib_enabled(&path));
    }
}
