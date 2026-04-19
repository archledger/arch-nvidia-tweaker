use std::path::Path;
use std::process::Command;

use crate::core::aur;
use crate::core::bootloader::{self, NVIDIA_DRM_PARAM};
use crate::core::gpu::{GpuInventory, GpuVendor, NvidiaGeneration};
use crate::core::hardware::FormFactor;
use crate::core::Context;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

impl Severity {
    pub fn marker(self) -> &'static str {
        match self {
            Self::Info => "[info]",
            Self::Warning => "[warn]",
            Self::Error => "[err ]",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: Severity,
    pub title: String,
    pub detail: String,
    pub fix_hint: Option<String>,
}

impl Finding {
    fn info(title: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            title: title.into(),
            detail: detail.into(),
            fix_hint: None,
        }
    }

    fn warn(title: impl Into<String>, detail: impl Into<String>, fix: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            title: title.into(),
            detail: detail.into(),
            fix_hint: Some(fix.into()),
        }
    }

    fn error(title: impl Into<String>, detail: impl Into<String>, fix: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            title: title.into(),
            detail: detail.into(),
            fix_hint: Some(fix.into()),
        }
    }
}

pub fn scan(ctx: &Context, gpus: &GpuInventory, form: FormFactor) -> Vec<Finding> {
    let mut findings = Vec::new();

    if gpus.gpus.is_empty() {
        findings.push(Finding::warn(
            "No display controller detected",
            "lspci returned no GPU devices.",
            "Install pciutils and verify `lspci` output manually.",
        ));
        return findings;
    }

    for g in &gpus.gpus {
        let driver = g.kernel_driver.as_deref().unwrap_or("none");
        findings.push(Finding::info(
            format!(
                "{:?} GPU {} at {}",
                g.vendor,
                g.display_name(),
                g.pci_address
            ),
            format!(
                "kernel driver: {driver}{}",
                g.nvidia_gen
                    .map(|gen| format!(" • {}", gen.human()))
                    .unwrap_or_default()
            ),
        ));
    }

    // Phase 15: informational / advisory findings for the detected driver family.
    if gpus.has_intel_xe() {
        findings.push(Finding::info(
            "Intel GPU on modern `xe` kernel driver",
            "The xe driver handles GuC/HuC firmware natively — no `i915.enable_guc=3` needed.",
        ));
    }
    if gpus.has_amd_radeon_legacy() {
        findings.push(Finding::warn(
            "AMD GPU on legacy `radeon` kernel driver",
            "Modern Arch kernels prefer the `amdgpu` driver for GCN 1.2+ hardware. The `radeon` \
             driver is correct for pre-GCN Terascale cards only.",
            "If you have a GCN 1.2+ GPU, check that amdgpu isn't disabled via a modprobe blacklist.",
        ));
    }

    let nvidia_present = gpus.has_nvidia();

    if !nvidia_present {
        findings.push(Finding::info(
            "No NVIDIA GPU — NVIDIA-specific actions will be skipped",
            "Generic gaming setup (multilib, Vulkan ICDs, gamemode, mangohud) is still applicable.",
        ));
    } else {
        check_cmdline_modeset(ctx, &mut findings);
        check_nvidia_package(gpus, &mut findings);
        check_nvidia_module_loaded(&mut findings);
        check_nouveau_conflict(&mut findings);
        check_suspend_services(&mut findings);
        check_mkinitcpio_modules(ctx, &mut findings);
        if gpus.is_hybrid() {
            check_prime_setup(&mut findings);
        }
    }

    check_multilib(ctx, &mut findings);
    check_vulkan_drivers(gpus, &mut findings);
    check_gaming_tools(&mut findings);
    check_user_video_group(&mut findings);
    check_session_type(&mut findings);
    check_vm_max_map_count(&mut findings);
    check_bumblebee_leftover(&mut findings);
    check_aur_helper(gpus, &mut findings);

    // Phase 15 + Phase 16 sanitation — surface through the diagnostics scanner so that
    // `--diagnose` CLI users see the same warnings the GUI banner shows.
    for w in crate::core::gaming::sanitation_warnings(gpus) {
        findings.push(Finding::warn(w.title, w.detail, w.remediation));
    }
    for w in crate::core::wayland::sanitation_warnings(ctx) {
        findings.push(Finding::warn(w.title(), w.detail(), w.remediation()));
    }

    // Phase 20: surface every RepairAction the self-heal scanner detected. These are
    // legacy artifacts from older archgpu versions that a state-aware probe (like
    // `wayland::check_state`) can't distinguish from intentional user config — only
    // knowledge of what archgpu USED to write on the current topology can. Severity is
    // Warning because the system keeps running, but fix is load-bearing (broken display
    // routing or driverless GPU).
    for action in crate::core::repair::scan(ctx, gpus, form) {
        findings.push(Finding::warn(
            "Legacy archgpu artifact detected",
            action.human_summary(),
            "Run `archgpu --apply-repair` or enable the Repair tweak in the GUI. Idempotent; backups land in /var/backups/archgpu/.",
        ));
    }

    findings
}

fn check_cmdline_modeset(ctx: &Context, out: &mut Vec<Finding>) {
    match bootloader::detect(ctx) {
        Ok(mgr) => {
            let desc = mgr.describe();
            match mgr.has_parameter(NVIDIA_DRM_PARAM) {
                Ok(true) => out.push(Finding::info(
                    format!("nvidia-drm.modeset=1 present — {desc}"),
                    "",
                )),
                Ok(false) => out.push(Finding::warn(
                    format!("nvidia-drm.modeset=1 missing — {desc}"),
                    "Required for NVIDIA on Wayland and for early KMS.",
                    "Run `archgpu --apply-bootloader` as root.",
                )),
                Err(e) => out.push(Finding::warn(
                    format!("Could not inspect bootloader cmdline ({desc})"),
                    format!("{e:#}"),
                    "Check cmdline source permissions.",
                )),
            }
        }
        Err(e) => out.push(Finding::warn(
            "Bootloader not detected",
            format!("{e}"),
            "Unsupported bootloader type — only GRUB / systemd-boot / Limine / UKI are handled.",
        )),
    }
}

fn check_nvidia_package(gpus: &GpuInventory, out: &mut Vec<Finding>) {
    let known = &[
        "nvidia",
        "nvidia-open",
        "nvidia-dkms",
        "nvidia-open-dkms",
        "nvidia-lts",
        "nvidia-lts-open",
        "nvidia-470xx-dkms",
        "nvidia-390xx-dkms",
    ];
    let Some(installed) = pacman_query_installed(known) else {
        out.push(Finding::info(
            "pacman not available — skipped NVIDIA package check",
            "",
        ));
        return;
    };
    if installed.is_empty() {
        let suggestion = gpus
            .primary_nvidia()
            .and_then(|g| g.recommended_nvidia_package())
            .map(|r| r.package)
            .unwrap_or("nvidia-open-dkms");
        out.push(Finding::error(
            "No NVIDIA driver package installed",
            "NVIDIA hardware present but no proprietary or open driver package found.",
            format!("Install `{suggestion}` (or use `--apply-gaming`)."),
        ));
    } else {
        out.push(Finding::info(
            format!("NVIDIA driver installed: {}", installed.join(", ")),
            "",
        ));
    }
}

fn check_nvidia_module_loaded(out: &mut Vec<Finding>) {
    let modules = std::fs::read_to_string("/proc/modules").unwrap_or_default();
    if modules.lines().any(|l| l.starts_with("nvidia ")) {
        out.push(Finding::info("nvidia kernel module loaded", ""));
    } else {
        out.push(Finding::warn(
            "nvidia kernel module NOT loaded",
            "The NVIDIA driver isn't active.",
            "Install an NVIDIA driver and reboot (or `modprobe nvidia`).",
        ));
    }
}

fn check_nouveau_conflict(out: &mut Vec<Finding>) {
    let modules = std::fs::read_to_string("/proc/modules").unwrap_or_default();
    let nvidia = modules.lines().any(|l| l.starts_with("nvidia "));
    let nouveau = modules.lines().any(|l| l.starts_with("nouveau "));
    if nvidia && nouveau {
        out.push(Finding::error(
            "nvidia and nouveau both loaded",
            "Conflicting drivers; system will be unstable.",
            "Blacklist nouveau via /etc/modprobe.d/blacklist-nouveau.conf and rebuild initramfs.",
        ));
    } else if nouveau {
        out.push(Finding::info(
            "nouveau is the active NVIDIA driver",
            "Proprietary/open kernel modules are not currently in use.",
        ));
    }
}

fn check_suspend_services(out: &mut Vec<Finding>) {
    let services = [
        "nvidia-suspend.service",
        "nvidia-hibernate.service",
        "nvidia-resume.service",
    ];
    let mut not_enabled = Vec::new();
    for svc in services {
        match Command::new("systemctl").args(["is-enabled", svc]).output() {
            Ok(o) => {
                let state = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if state != "enabled" {
                    not_enabled.push(format!("{svc}={state}"));
                }
            }
            Err(_) => return,
        }
    }
    if !not_enabled.is_empty() {
        out.push(Finding::warn(
            "NVIDIA suspend/resume services not all enabled",
            not_enabled.join(", "),
            "Run `archgpu --apply-power` as root.",
        ));
    } else {
        out.push(Finding::info("NVIDIA suspend/resume services enabled", ""));
    }
}

fn check_mkinitcpio_modules(ctx: &Context, out: &mut Vec<Finding>) {
    let dropin = ctx.paths.mkinitcpio_d.join("nvidia-modules.conf");
    if dropin.exists() {
        out.push(Finding::info(
            "mkinitcpio drop-in for NVIDIA modules present",
            dropin.display().to_string(),
        ));
        return;
    }
    let main_conf = Path::new("/etc/mkinitcpio.conf");
    if let Ok(body) = std::fs::read_to_string(main_conf) {
        let has_nvidia = body.lines().any(|l| {
            let t = l.trim();
            !t.starts_with('#') && t.contains("MODULES=") && t.contains("nvidia")
        });
        if has_nvidia {
            out.push(Finding::info(
                "NVIDIA modules in /etc/mkinitcpio.conf MODULES",
                "",
            ));
        } else {
            out.push(Finding::warn(
                "NVIDIA modules not in initramfs",
                "nvidia/nvidia_modeset/nvidia_uvm/nvidia_drm should be in MODULES for early KMS.",
                "Run `archgpu --apply-wayland`.",
            ));
        }
    }
}

fn check_prime_setup(out: &mut Vec<Finding>) {
    let Some(installed) = pacman_query_installed(&["nvidia-prime"]) else {
        return;
    };
    if installed.is_empty() {
        out.push(Finding::warn(
            "Hybrid GPU detected but `nvidia-prime` not installed",
            "Provides the `prime-run` wrapper used to launch apps on the NVIDIA dGPU.",
            "pacman -S --needed nvidia-prime",
        ));
    } else {
        out.push(Finding::info(
            "nvidia-prime installed",
            "`prime-run` is available.",
        ));
    }
}

fn check_multilib(ctx: &Context, out: &mut Vec<Finding>) {
    if crate::core::gaming::is_multilib_enabled(&ctx.paths.pacman_conf) {
        out.push(Finding::info("[multilib] repo enabled", ""));
    } else {
        out.push(Finding::warn(
            "[multilib] repo not enabled",
            "32-bit Steam/Wine libraries (lib32-*) will be unavailable.",
            "Run `archgpu --apply-gaming`.",
        ));
    }
}

fn check_vulkan_drivers(gpus: &GpuInventory, out: &mut Vec<Finding>) {
    let Some(installed) = pacman_query_installed(&[
        "vulkan-icd-loader",
        "lib32-vulkan-icd-loader",
        "vulkan-intel",
        "lib32-vulkan-intel",
        "vulkan-radeon",
        "lib32-vulkan-radeon",
        "nvidia-utils",
        "lib32-nvidia-utils",
    ]) else {
        return;
    };

    let missing = |pkg: &str| !installed.iter().any(|p| p == pkg);
    let mut absent = Vec::new();
    for pkg in ["vulkan-icd-loader", "lib32-vulkan-icd-loader"] {
        if missing(pkg) {
            absent.push(pkg);
        }
    }
    if !absent.is_empty() {
        out.push(Finding::warn(
            "Vulkan loader missing",
            format!("missing: {}", absent.join(", ")),
            format!("pacman -S --needed {}", absent.join(" ")),
        ));
    }

    for g in &gpus.gpus {
        let (pkg64, pkg32) = match g.vendor {
            GpuVendor::Nvidia => ("nvidia-utils", "lib32-nvidia-utils"),
            GpuVendor::Amd => ("vulkan-radeon", "lib32-vulkan-radeon"),
            GpuVendor::Intel => ("vulkan-intel", "lib32-vulkan-intel"),
            GpuVendor::Other => continue,
        };
        let mut needed = Vec::new();
        if missing(pkg64) {
            needed.push(pkg64);
        }
        if missing(pkg32) {
            needed.push(pkg32);
        }
        if !needed.is_empty() {
            out.push(Finding::warn(
                format!("Vulkan driver missing for {:?}", g.vendor),
                format!("missing: {}", needed.join(", ")),
                format!("pacman -S --needed {}", needed.join(" ")),
            ));
        }
    }
}

fn check_gaming_tools(out: &mut Vec<Finding>) {
    let Some(installed) =
        pacman_query_installed(&["gamemode", "lib32-gamemode", "mangohud", "lib32-mangohud"])
    else {
        return;
    };
    let missing: Vec<_> = ["gamemode", "lib32-gamemode", "mangohud", "lib32-mangohud"]
        .iter()
        .filter(|p| !installed.iter().any(|i| i == *p))
        .copied()
        .collect();
    if missing.is_empty() {
        out.push(Finding::info(
            "gamemode + mangohud installed (64 + 32-bit)",
            "",
        ));
    } else {
        out.push(Finding::warn(
            "Recommended gaming tools missing",
            format!("missing: {}", missing.join(", ")),
            "Run `archgpu --apply-gaming`.",
        ));
    }
}

fn check_user_video_group(out: &mut Vec<Finding>) {
    let Ok(got) = Command::new("groups").output() else {
        return;
    };
    let groups = String::from_utf8_lossy(&got.stdout);
    if groups.split_whitespace().any(|g| g == "video") {
        out.push(Finding::info("current user in `video` group", ""));
    } else {
        out.push(Finding::info(
            "current user NOT in `video` group",
            "Some GPU utilities (brightness, NVIDIA control via non-X backends) need this.",
        ));
    }
}

fn check_session_type(out: &mut Vec<Finding>) {
    let session = std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "unknown".to_string());
    out.push(Finding::info(
        format!("XDG_SESSION_TYPE = {session}"),
        if session == "wayland" {
            "NVIDIA Wayland requires driver ≥ 545 and all four kernel modules loaded early."
                .to_string()
        } else {
            String::new()
        },
    ));
}

fn check_vm_max_map_count(out: &mut Vec<Finding>) {
    let Ok(body) = std::fs::read_to_string("/proc/sys/vm/max_map_count") else {
        return;
    };
    let value: u64 = body.trim().parse().unwrap_or(0);
    const TARGET: u64 = 1_048_576;
    if value >= TARGET {
        out.push(Finding::info(
            format!("vm.max_map_count = {value}"),
            "Sufficient for modern games (Star Citizen, Hogwarts Legacy, etc.).",
        ));
    } else {
        out.push(Finding::warn(
            format!("vm.max_map_count is low ({value})"),
            "Some games need ≥ 1048576 to avoid crashes/allocation errors.",
            "Run `archgpu --apply-gaming` to write /etc/sysctl.d/99-gaming.conf.",
        ));
    }
}

fn check_bumblebee_leftover(out: &mut Vec<Finding>) {
    let indicators: &[&Path] = &[
        Path::new("/etc/bumblebee"),
        Path::new("/usr/bin/optirun"),
        Path::new("/usr/bin/bumblebeed"),
    ];
    let present: Vec<_> = indicators
        .iter()
        .filter(|p| p.exists())
        .map(|p| p.display().to_string())
        .collect();
    if !present.is_empty() {
        out.push(Finding::warn(
            "Bumblebee residue detected",
            format!("found: {}", present.join(", ")),
            "Bumblebee is deprecated on Arch — migrate to PRIME render offload.",
        ));
    }
}

fn check_aur_helper(gpus: &GpuInventory, out: &mut Vec<Finding>) {
    let needs_aur = gpus.primary_nvidia().is_some_and(|nv| {
        matches!(
            nv.nvidia_gen,
            Some(NvidiaGeneration::Maxwell)
                | Some(NvidiaGeneration::Kepler)
                | Some(NvidiaGeneration::Fermi)
        )
    });

    match aur::detect_helper() {
        Some(h) => out.push(Finding::info(
            format!("AUR helper available: {}", h.name()),
            "",
        )),
        None => {
            if needs_aur {
                out.push(Finding::warn(
                    "No AUR helper and legacy NVIDIA driver required",
                    "Maxwell/Kepler/Fermi GPUs need nvidia-470xx-dkms or nvidia-390xx-dkms from AUR.",
                    "Run `archgpu --apply-gaming` — yay-bin will be bootstrapped automatically.",
                ));
            } else {
                out.push(Finding::info(
                    "No AUR helper installed (yay / paru)",
                    "Not required for this host's driver path.",
                ));
            }
        }
    }
}

fn pacman_query_installed(names: &[&str]) -> Option<Vec<String>> {
    let out = Command::new("pacman").arg("-Qq").output().ok()?;
    if !out.status.success() {
        return Some(Vec::new());
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let set: std::collections::HashSet<&str> = names.iter().copied().collect();
    Some(
        body.lines()
            .filter(|l| set.contains(l.trim()))
            .map(|l| l.trim().to_string())
            .collect(),
    )
}
