use anyhow::{Context as _, Result};
use std::process::Command;

use crate::core::gpu::GpuInventory;
use crate::core::hardware::FormFactor;
use crate::core::state::TweakState;
use crate::core::Context;
use crate::utils::fs_helper::{write_dropin, ChangeReport};

const MODPROBE_FILE: &str = "zzz-nvidia-tweaks-auto.conf";
const NOUVEAU_BLACKLIST_FILE: &str = "blacklist-nouveau.conf";

const NOUVEAU_BLACKLIST_CONTENT: &str = "\
# Managed by archgpu — prevents nouveau from loading while the proprietary (or
# open GSP) NVIDIA kernel modules are active. Remove this file to allow nouveau again.
blacklist nouveau
options nouveau modeset=0
";

const SYSTEMD_UNITS: &[&str] = &[
    "nvidia-suspend.service",
    "nvidia-hibernate.service",
    "nvidia-resume.service",
];

/// Returns `Applied` only when BOTH the modprobe drop-in exists AND `nvidia-suspend.service`
/// is `enabled` per `systemctl is-enabled`. Incompatible on non-NVIDIA hosts.
///
/// `systemctl is-enabled` is a real system probe — in unit-test contexts with no such unit,
/// it returns "disabled"/"not-found" so the function yields Unapplied.
pub fn check_state(ctx: &Context, gpus: &GpuInventory) -> TweakState {
    if !gpus.has_nvidia() {
        return TweakState::Incompatible;
    }
    let modprobe = ctx.paths.modprobe_d.join(MODPROBE_FILE);
    if !modprobe.exists() {
        return TweakState::Unapplied;
    }
    let enabled = Command::new("systemctl")
        .args(["is-enabled", "nvidia-suspend.service"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "enabled")
        .unwrap_or(false);
    if enabled {
        // systemctl reports the unit enabled AND the modprobe drop-in is on disk. The
        // option itself (NVreg_UseKernelSuspendNotifiers, NVreg_DynamicPowerManagement)
        // is picked up on the next modprobe/boot; since we don't probe
        // /sys/module/nvidia/parameters/<p> here, Active is the honest answer.
        TweakState::Active
    } else {
        TweakState::Unapplied
    }
}

pub fn apply(ctx: &Context, form: FormFactor) -> Result<Vec<ChangeReport>> {
    let mut reports = Vec::new();

    let content = modprobe_content(form);
    let target = ctx.paths.modprobe_d.join(MODPROBE_FILE);
    reports.push(write_dropin(
        &target,
        &content,
        &ctx.paths.backup_dir,
        ctx.mode.is_dry_run(),
    )?);

    // nouveau blacklist (outer gating in run_actions ensures this only runs on NVIDIA hosts)
    let nouveau_target = ctx.paths.modprobe_d.join(NOUVEAU_BLACKLIST_FILE);
    reports.push(write_dropin(
        &nouveau_target,
        NOUVEAU_BLACKLIST_CONTENT,
        &ctx.paths.backup_dir,
        ctx.mode.is_dry_run(),
    )?);

    for unit in SYSTEMD_UNITS {
        reports.push(enable_unit(ctx, unit)?);
    }

    Ok(reports)
}

pub fn modprobe_content(form: FormFactor) -> String {
    let mut s = String::new();
    s.push_str("# Managed by archgpu — do not edit by hand.\n");
    s.push_str("# Required on all hosts to avoid Wayland wake artifacts.\n");
    s.push_str("options nvidia NVreg_UseKernelSuspendNotifiers=1\n");
    if form == FormFactor::Laptop {
        s.push_str("# Hybrid-graphics dynamic power management (laptops only).\n");
        s.push_str("options nvidia NVreg_DynamicPowerManagement=0x02\n");
    }
    s
}

fn enable_unit(ctx: &Context, unit: &str) -> Result<ChangeReport> {
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned {
            detail: format!("systemctl enable {unit}"),
        });
    }

    let status = Command::new("systemctl")
        .arg("is-enabled")
        .arg(unit)
        .output()
        .with_context(|| format!("checking systemctl is-enabled {unit}"))?;
    let out = String::from_utf8_lossy(&status.stdout);
    if out.trim() == "enabled" {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!("{unit} already enabled"),
        });
    }

    log::info!("Enabling {unit}");
    let exit = Command::new("systemctl")
        .arg("enable")
        .arg(unit)
        .status()
        .with_context(|| format!("spawning systemctl enable {unit}"))?;
    if !exit.success() {
        anyhow::bail!("systemctl enable {unit} exited with {exit}");
    }
    Ok(ChangeReport::Applied {
        detail: format!("enabled {unit}"),
        backup: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laptop_content_has_both_options() {
        let s = modprobe_content(FormFactor::Laptop);
        assert!(s.contains("NVreg_UseKernelSuspendNotifiers=1"));
        assert!(s.contains("NVreg_DynamicPowerManagement=0x02"));
    }

    #[test]
    fn desktop_content_omits_laptop_option() {
        let s = modprobe_content(FormFactor::Desktop);
        assert!(s.contains("NVreg_UseKernelSuspendNotifiers=1"));
        assert!(!s.contains("NVreg_DynamicPowerManagement"));
    }

    #[test]
    fn unknown_form_treated_as_non_laptop() {
        let s = modprobe_content(FormFactor::Unknown);
        assert!(!s.contains("NVreg_DynamicPowerManagement"));
    }

    #[test]
    fn nouveau_blacklist_content_is_stable() {
        assert!(NOUVEAU_BLACKLIST_CONTENT.contains("blacklist nouveau"));
        assert!(NOUVEAU_BLACKLIST_CONTENT.contains("options nouveau modeset=0"));
    }
}
