pub mod aur;
pub mod auto;
pub mod bootloader;
pub mod cpu;
pub mod diagnostics;
pub mod gaming;
pub mod gpu;
pub mod hardware;
pub mod nvidia;
pub mod power;
pub mod prime;
pub mod rendering;
pub mod repair;
pub mod state;
pub mod wayland;

#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;

use crate::core::hardware::FormFactor;
use crate::utils::fs_helper::ChangeReport;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Apply,
    DryRun,
}

impl ExecutionMode {
    pub fn is_dry_run(self) -> bool {
        matches!(self, Self::DryRun)
    }
}

#[derive(Debug, Clone)]
pub struct SystemPaths {
    pub profile_d: PathBuf,
    pub mkinitcpio_d: PathBuf,
    pub modprobe_d: PathBuf,
    pub sysctl_d: PathBuf,
    pub xorg_d: PathBuf,
    pub etc_x11_xorg_conf: PathBuf, // Phase 16: legacy-xorg.conf probe target
    pub kernel_cmdline: PathBuf,
    pub pacman_conf: PathBuf,
    pub dmi_chassis: PathBuf,
    pub cpuinfo: PathBuf, // Phase 21: CET-IBT probe via /proc/cpuinfo flags
    // Phase 22: smart-diagnostics probes for software-rendering root-causes.
    pub kernel_osrelease: PathBuf,       // /proc/sys/kernel/osrelease — running kernel version
    pub modules_dir: PathBuf,            // /usr/lib/modules — each running kernel has a subdir
    pub proc_cmdline: PathBuf,           // /proc/cmdline — live kernel cmdline (for nomodeset check)
    pub secureboot_efivars_dir: PathBuf, // /sys/firmware/efi/efivars — SecureBoot-<guid> variable lives here
    pub vulkan_icd_dir: PathBuf,         // /usr/share/vulkan/icd.d — ICD JSONs referencing driver .so
    pub backup_dir: PathBuf,
    // Phase 17: live-kernel-state probe. `sys_module` is the root of the /sys/module tree;
    // specific params live under <sys_module>/<module>/parameters/<name>. A missing
    // <sys_module>/<module>/ directory means the kernel module isn't loaded, which is the
    // same signal as "module loaded but parameter reports wrong value" for our purposes.
    pub sys_module: PathBuf,
    // Phase 11: multi-bootloader paths
    pub grub_default: PathBuf,
    pub grub_cfg: PathBuf,
    pub sdb_loader_conf: PathBuf,
    pub sdb_entries: PathBuf,
    pub limine_candidates: Vec<PathBuf>,
}

impl SystemPaths {
    pub fn production() -> Self {
        Self {
            profile_d: PathBuf::from("/etc/profile.d"),
            mkinitcpio_d: PathBuf::from("/etc/mkinitcpio.conf.d"),
            modprobe_d: PathBuf::from("/etc/modprobe.d"),
            sysctl_d: PathBuf::from("/etc/sysctl.d"),
            xorg_d: PathBuf::from("/etc/X11/xorg.conf.d"),
            etc_x11_xorg_conf: PathBuf::from("/etc/X11/xorg.conf"),
            kernel_cmdline: PathBuf::from("/etc/kernel/cmdline"),
            pacman_conf: PathBuf::from("/etc/pacman.conf"),
            dmi_chassis: PathBuf::from("/sys/class/dmi/id/chassis_type"),
            cpuinfo: PathBuf::from("/proc/cpuinfo"),
            kernel_osrelease: PathBuf::from("/proc/sys/kernel/osrelease"),
            modules_dir: PathBuf::from("/usr/lib/modules"),
            proc_cmdline: PathBuf::from("/proc/cmdline"),
            secureboot_efivars_dir: PathBuf::from("/sys/firmware/efi/efivars"),
            vulkan_icd_dir: PathBuf::from("/usr/share/vulkan/icd.d"),
            backup_dir: PathBuf::from("/var/backups/archgpu"),
            sys_module: PathBuf::from("/sys/module"),
            grub_default: PathBuf::from("/etc/default/grub"),
            grub_cfg: PathBuf::from("/boot/grub/grub.cfg"),
            sdb_loader_conf: PathBuf::from("/boot/loader/loader.conf"),
            sdb_entries: PathBuf::from("/boot/loader/entries"),
            limine_candidates: vec![
                PathBuf::from("/boot/limine.conf"),
                PathBuf::from("/boot/limine/limine.conf"),
                PathBuf::from("/boot/efi/limine.conf"),
            ],
        }
    }
}

#[cfg(test)]
impl SystemPaths {
    pub fn rooted(root: &Path) -> Self {
        Self {
            profile_d: root.join("etc/profile.d"),
            mkinitcpio_d: root.join("etc/mkinitcpio.conf.d"),
            modprobe_d: root.join("etc/modprobe.d"),
            sysctl_d: root.join("etc/sysctl.d"),
            xorg_d: root.join("etc/X11/xorg.conf.d"),
            etc_x11_xorg_conf: root.join("etc/X11/xorg.conf"),
            kernel_cmdline: root.join("etc/kernel/cmdline"),
            pacman_conf: root.join("etc/pacman.conf"),
            dmi_chassis: root.join("sys/class/dmi/id/chassis_type"),
            cpuinfo: root.join("proc/cpuinfo"),
            kernel_osrelease: root.join("proc/sys/kernel/osrelease"),
            modules_dir: root.join("usr/lib/modules"),
            proc_cmdline: root.join("proc/cmdline"),
            secureboot_efivars_dir: root.join("sys/firmware/efi/efivars"),
            vulkan_icd_dir: root.join("usr/share/vulkan/icd.d"),
            backup_dir: root.join("var/backups/archgpu"),
            sys_module: root.join("sys/module"),
            grub_default: root.join("etc/default/grub"),
            grub_cfg: root.join("boot/grub/grub.cfg"),
            sdb_loader_conf: root.join("boot/loader/loader.conf"),
            sdb_entries: root.join("boot/loader/entries"),
            limine_candidates: vec![
                root.join("boot/limine.conf"),
                root.join("boot/limine/limine.conf"),
                root.join("boot/efi/limine.conf"),
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct Context {
    pub paths: SystemPaths,
    pub mode: ExecutionMode,
}

impl Context {
    pub fn production(mode: ExecutionMode) -> Self {
        Self {
            paths: SystemPaths::production(),
            mode,
        }
    }
}

#[cfg(test)]
impl Context {
    pub fn rooted_for_test(root: &Path, mode: ExecutionMode) -> Self {
        Self {
            paths: SystemPaths::rooted(root),
            mode,
        }
    }
}

/// The set of apply-actions that the caller (CLI or GUI) wants to run.
#[derive(Debug, Default, Clone, Copy)]
pub struct Actions {
    pub wayland: bool,
    pub bootloader: bool,
    pub power: bool,
    pub gaming: bool,
    /// Phase 20: self-heal pass. Deletes legacy PRIME drop-ins on desktop hybrids,
    /// removes `nvidia-prime` on desktop hybrids, forces DKMS rebuild when the module
    /// failed to build. Runs BEFORE the other actions so their state probes see a
    /// clean system. Idempotent — on a healthy host it's a no-op.
    pub repair: bool,
}

impl Actions {
    pub fn all() -> Self {
        Self {
            wayland: true,
            bootloader: true,
            power: true,
            gaming: true,
            repair: true,
        }
    }

    pub fn any(&self) -> bool {
        self.wayland || self.bootloader || self.power || self.gaming || self.repair
    }
}

/// Execute the selected actions in order, gating NVIDIA-specific actions on the inventory.
/// Subprocess output (pacman, mkinitcpio, makepkg, sudo-u yay) is streamed line-by-line into
/// `progress` so CLI/GUI can surface it live.
pub fn run_actions(
    ctx: &Context,
    form: FormFactor,
    gpus: &gpu::GpuInventory,
    actions: Actions,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<(&'static str, ChangeReport)>> {
    let mut out = Vec::new();
    let has_nv = gpus.has_nvidia();

    // Phase 20: run repair FIRST. Every subsequent action's `check_state` probe is
    // invalidated if a pre-Phase-19 PRIME drop-in or a broken DKMS module is still on
    // disk — clean those up before the rest of the pipeline evaluates state.
    if actions.repair {
        for r in repair::apply(ctx, gpus, form, assume_yes, progress)? {
            out.push(("repair", r));
        }
    }

    if actions.wayland {
        if has_nv {
            for r in wayland::apply(ctx, gpus, form)? {
                out.push(("wayland", r));
            }
        } else {
            out.push((
                "wayland",
                skip_non_nvidia("Wayland env + mkinitcpio modules + PRIME"),
            ));
        }
    }
    if actions.bootloader {
        // Phase 15/16: the bootloader action now covers multi-vendor kernel cmdline tweaks
        // (NVIDIA modeset+fbdev, amdgpu.ppfeaturemask, i915.enable_guc). It applies to any
        // host whose GPU inventory yields a non-empty `required_kernel_params` list —
        // Intel-xe-only hosts cleanly report "no cmdline params needed".
        out.push(("bootloader", bootloader::apply(ctx, gpus, progress)?));
    }
    if actions.power {
        if has_nv {
            for r in power::apply(ctx, form)? {
                out.push(("power", r));
            }
        } else {
            out.push((
                "power",
                skip_non_nvidia("nvidia suspend/resume + modprobe + nouveau blacklist"),
            ));
        }
    }
    if actions.gaming {
        for r in gaming::apply(ctx, gpus, form, assume_yes, progress)? {
            out.push(("gaming", r));
        }
    }
    Ok(out)
}

fn skip_non_nvidia(what: &str) -> ChangeReport {
    ChangeReport::AlreadyApplied {
        detail: format!("skipped {what}: no NVIDIA GPU on this host"),
    }
}
