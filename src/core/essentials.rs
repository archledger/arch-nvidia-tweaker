//! Phase 26 — essential userspace that archgpu should guarantee on every Arch system,
//! independent of gaming. Covers:
//!   - Vulkan loader (cross-vendor) + vendor ICDs
//!   - Mesa GL (cross-vendor) — required by everything that draws 3D
//!   - Split firmware packages `linux-firmware-amdgpu` / `linux-firmware-intel`
//!     (carved out as separately-installable splits in the 2025 reorg).
//!     `linux-firmware` still exists but is now a meta-package that depends on
//!     the splits — so `pacman -S linux-firmware` installs everything via deps.
//!     Installing the specific splits directly is stricter (guarantees the
//!     exact GPU-vendor firmware is present even if `linux-firmware` is
//!     manually uninstalled) and documents the dependency relationship.
//!   - Video acceleration: `intel-media-driver` (Gen 8+) or `libva-intel-driver`
//!     (Gen 6/7 — Sandy Bridge / Ivy Bridge / Haswell)
//!   - Diagnostic userspace: `vulkan-tools` (vulkaninfo/vkcube), `clinfo` (OpenCL
//!     platforms), `libva-utils` (vainfo), `vdpauinfo`
//!
//! Deliberately OMITTED from this list:
//!   - `gamemode` / `mangohud` / `goverlay` — gaming-specific; stay in `gaming.rs`
//!   - NVIDIA driver packages — stay in `gaming.rs` to avoid double-install
//!     (essentials is vendor-agnostic base, gaming layers the full NVIDIA stack)
//!   - Compute runtimes (ROCm / intel-compute-runtime / CUDA) — opt-in only,
//!     surfaced as Info-severity advisories by `diagnostics::scan`
//!
//! Intel VA-API routing. The `intel-media-driver` (iHD) package supports Broadwell
//! (Gen 8) and newer. `libva-intel-driver` (i965) covers Gen 4 through Gen 9 but
//! is functionally unmaintained (last upstream release 2.4.1). We route the
//! installed package by PCI device ID:
//!   - device_id < 0x1600  → Gen 7 or older → libva-intel-driver
//!   - device_id >= 0x1600 → Gen 8 or newer → intel-media-driver
//!
//! 0x1600 is the first Broadwell (Gen 8) device-ID prefix. This is a heuristic;
//! users with unusual hardware can override by editing the install manually.

use anyhow::Result;
use std::collections::HashSet;
use std::process::Command;

use crate::core::gpu::{GpuInfo, GpuInventory, GpuVendor};
use crate::core::state::TweakState;
use crate::core::{Context, ExecutionMode};
use crate::utils::fs_helper::ChangeReport;
use crate::utils::process::run_streaming;

/// Cross-vendor base always installed, regardless of which GPU is present. These
/// are the minimum userspace pieces for any working Linux graphics stack:
/// the Vulkan loader (every Vulkan app links against it) + the four diagnostic
/// tools used by archgpu's own `--diagnose` probes and by users themselves.
pub const ALWAYS_ON_PACKAGES: &[&str] = &[
    "vulkan-icd-loader",
    "lib32-vulkan-icd-loader",
    "vulkan-tools",
    "clinfo",
    "libva-utils",
    "vdpauinfo",
];

/// First Broadwell (Intel Gen 8) PCI device-ID prefix. See module doc for rationale.
const INTEL_IHD_MIN_DEVICE_ID: u16 = 0x1600;

/// Compute the essentials package set for a given GPU inventory.
pub fn resolve_packages(gpus: &GpuInventory) -> Vec<String> {
    let mut pkgs: Vec<String> = ALWAYS_ON_PACKAGES.iter().map(|s| s.to_string()).collect();
    let mut add = |pkg: &str| {
        let s = pkg.to_string();
        if !pkgs.contains(&s) {
            pkgs.push(s);
        }
    };

    let has_amd = gpus.has_amd();
    let has_intel = gpus.has_intel();

    // Mesa GL is cross-vendor — it's the GL/VA-API/VDPAU implementation for
    // amdgpu/radeon + i915/xe + nouveau. Even NVIDIA-only hosts benefit from
    // having it as a software-render fallback (`lavapipe` path). Skip only
    // when no open-source-driven GPU exists AND NVIDIA is the sole vendor;
    // in practice we keep mesa for universality.
    if has_amd || has_intel {
        add("mesa");
        add("lib32-mesa");
    }

    if has_amd {
        add("vulkan-radeon");
        add("lib32-vulkan-radeon");
        // Split firmware — carved out of `linux-firmware` in 2025. Required for
        // amdgpu to load correctly on any GCN1.2+ card.
        add("linux-firmware-amdgpu");
    }

    if has_intel {
        add("vulkan-intel");
        add("lib32-vulkan-intel");
        // Split firmware — same 2025 split as AMD. Provides GuC/HuC blobs i915/xe need.
        add("linux-firmware-intel");
        // Route VA-API driver by generation of the primary Intel GPU.
        if gpus.any_pre_broadwell_intel() {
            add("libva-intel-driver");
        } else {
            add("intel-media-driver");
        }
    }

    pkgs
}

/// Essentials state is Active when every package `resolve_packages` returns is installed.
/// Never Incompatible — the always-on Vulkan loader + diagnostic tools apply to every host.
pub fn check_state(ctx: &Context, gpus: &GpuInventory) -> TweakState {
    let _ = ctx; // reserved for future file-based checks; keeps signature parallel to peers
    let expected = resolve_packages(gpus);
    let Some(installed) = pacman_query_installed_set(&expected) else {
        // pacman unavailable — can't verify; leave the toggle callable.
        return TweakState::Unapplied;
    };
    if expected.iter().all(|p| installed.contains(p.as_str())) {
        TweakState::Active
    } else {
        TweakState::Unapplied
    }
}

pub fn apply(
    ctx: &Context,
    gpus: &GpuInventory,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ChangeReport>> {
    let pkgs = resolve_packages(gpus);
    if pkgs.is_empty() {
        return Ok(vec![ChangeReport::AlreadyApplied {
            detail: "no essential packages needed for this host".into(),
        }]);
    }
    let detail = format!("pacman -S --needed {}", pkgs.join(" "));
    if ctx.mode.is_dry_run() {
        return Ok(vec![ChangeReport::Planned { detail }]);
    }
    if matches!(ctx.mode, ExecutionMode::Apply) {
        let mut cmd = Command::new("pacman");
        cmd.args(["-S", "--needed"]);
        if assume_yes {
            cmd.arg("--noconfirm");
        }
        for p in &pkgs {
            cmd.arg(p);
        }
        progress(&format!("[pacman] {detail}"));
        let status = run_streaming(cmd, |line| progress(&format!("[pacman] {line}")))?;
        if !status.success() {
            anyhow::bail!("pacman -S --needed (essentials) exited with {status}");
        }
    }
    Ok(vec![ChangeReport::Applied {
        detail,
        backup: None,
    }])
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

/// Helper: does any Intel GPU in the inventory predate Broadwell?
///
/// Lives on `GpuInventory` for symmetry but implemented here to keep the
/// generation heuristic colocated with the package routing that depends on it.
pub(crate) trait IntelGenerationCheck {
    fn any_pre_broadwell_intel(&self) -> bool;
}

impl IntelGenerationCheck for GpuInventory {
    fn any_pre_broadwell_intel(&self) -> bool {
        self.gpus.iter().any(is_pre_broadwell_intel)
    }
}

fn is_pre_broadwell_intel(g: &GpuInfo) -> bool {
    g.vendor == GpuVendor::Intel && g.device_id < INTEL_IHD_MIN_DEVICE_ID
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::{GpuInfo, GpuVendor, NvidiaGeneration};

    fn gpu(vendor: GpuVendor, device_id: u16) -> GpuInfo {
        GpuInfo {
            vendor,
            vendor_id: match vendor {
                GpuVendor::Nvidia => 0x10de,
                GpuVendor::Amd => 0x1002,
                GpuVendor::Intel => 0x8086,
                GpuVendor::Other => 0,
            },
            device_id,
            pci_address: "0000:00:02.0".into(),
            vendor_name: format!("{vendor:?}"),
            product_name: "test".into(),
            kernel_driver: None,
            is_integrated: true,
            nvidia_gen: if vendor == GpuVendor::Nvidia {
                Some(NvidiaGeneration::Ampere)
            } else {
                None
            },
        }
    }

    fn inv(gpus: Vec<GpuInfo>) -> GpuInventory {
        GpuInventory { gpus }
    }

    #[test]
    fn always_on_packages_present_even_on_nvidia_only_host() {
        let pkgs = resolve_packages(&inv(vec![gpu(GpuVendor::Nvidia, 0x2204)]));
        for p in ALWAYS_ON_PACKAGES {
            assert!(pkgs.contains(&p.to_string()), "missing always-on {p}");
        }
        // NVIDIA-only should NOT pull mesa/firmware-amdgpu/firmware-intel —
        // essentials is vendor-agnostic base, not a driver installer.
        assert!(!pkgs.contains(&"linux-firmware-amdgpu".to_string()));
        assert!(!pkgs.contains(&"linux-firmware-intel".to_string()));
        assert!(!pkgs.contains(&"vulkan-radeon".to_string()));
        assert!(!pkgs.contains(&"vulkan-intel".to_string()));
    }

    #[test]
    fn amd_host_gets_split_firmware_and_mesa_stack() {
        let pkgs = resolve_packages(&inv(vec![gpu(GpuVendor::Amd, 0x73bf)]));
        assert!(pkgs.contains(&"mesa".to_string()));
        assert!(pkgs.contains(&"lib32-mesa".to_string()));
        assert!(pkgs.contains(&"vulkan-radeon".to_string()));
        assert!(pkgs.contains(&"lib32-vulkan-radeon".to_string()));
        assert!(pkgs.contains(&"linux-firmware-amdgpu".to_string()));
        // Defunct package that an earlier version of this tool installed — must NOT come back.
        assert!(!pkgs.contains(&"libva-mesa-driver".to_string()));
    }

    #[test]
    fn intel_gen12_host_gets_intel_media_driver() {
        // Lunar Lake (Intel Arc 140V) — the dev host.
        let pkgs = resolve_packages(&inv(vec![gpu(GpuVendor::Intel, 0x64a0)]));
        assert!(pkgs.contains(&"intel-media-driver".to_string()));
        assert!(!pkgs.contains(&"libva-intel-driver".to_string()));
        assert!(pkgs.contains(&"linux-firmware-intel".to_string()));
    }

    #[test]
    fn intel_ivy_bridge_host_gets_libva_intel_driver() {
        // Ivy Bridge HD 4000 — device ID 0x0166 < 0x1600 → i965 path.
        let pkgs = resolve_packages(&inv(vec![gpu(GpuVendor::Intel, 0x0166)]));
        assert!(pkgs.contains(&"libva-intel-driver".to_string()));
        assert!(!pkgs.contains(&"intel-media-driver".to_string()));
    }

    #[test]
    fn intel_broadwell_is_on_the_modern_boundary() {
        // Broadwell HD Graphics 5500 — device ID 0x1616; exactly at the boundary.
        let pkgs = resolve_packages(&inv(vec![gpu(GpuVendor::Intel, 0x1616)]));
        assert!(pkgs.contains(&"intel-media-driver".to_string()));
    }

    #[test]
    fn hybrid_intel_plus_nvidia_routes_by_intel_generation() {
        let pkgs = resolve_packages(&inv(vec![
            gpu(GpuVendor::Intel, 0x3e9b), // Coffee Lake UHD 630
            gpu(GpuVendor::Nvidia, 0x25a2),
        ]));
        assert!(pkgs.contains(&"intel-media-driver".to_string()));
        assert!(pkgs.contains(&"linux-firmware-intel".to_string()));
        // Hybrid gets Intel essentials but NOT NVIDIA-specific drivers
        // (those belong in `gaming.rs` / `wayland.rs`).
        assert!(pkgs.contains(&"mesa".to_string()));
    }

    #[test]
    fn resolve_does_not_duplicate_cross_vendor_mesa() {
        let pkgs = resolve_packages(&inv(vec![
            gpu(GpuVendor::Amd, 0x73bf),
            gpu(GpuVendor::Intel, 0x64a0),
        ]));
        let mesa_count = pkgs.iter().filter(|p| *p == "mesa").count();
        let lib32_mesa_count = pkgs.iter().filter(|p| *p == "lib32-mesa").count();
        assert_eq!(mesa_count, 1);
        assert_eq!(lib32_mesa_count, 1);
    }

    #[test]
    fn pre_broadwell_detector_matches_only_old_intel() {
        assert!(is_pre_broadwell_intel(&gpu(GpuVendor::Intel, 0x0166))); // Ivy Bridge
        assert!(is_pre_broadwell_intel(&gpu(GpuVendor::Intel, 0x0416))); // Haswell
        assert!(!is_pre_broadwell_intel(&gpu(GpuVendor::Intel, 0x1616))); // Broadwell
        assert!(!is_pre_broadwell_intel(&gpu(GpuVendor::Intel, 0x64a0))); // Lunar Lake
        assert!(!is_pre_broadwell_intel(&gpu(GpuVendor::Amd, 0x0166))); // AMD (same id, wrong vendor)
    }
}
