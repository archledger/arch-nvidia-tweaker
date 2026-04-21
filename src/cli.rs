use anyhow::Result;
use clap::Parser;

use crate::core::bootloader::{self, NVIDIA_DRM_PARAM};
use crate::core::diagnostics::{self, Finding, Severity};
use crate::core::gpu::{GpuInventory, PackageSource};
use crate::core::hardware::{self, FormFactor};
use crate::core::{self, Actions, Context, ExecutionMode};

#[derive(Debug, Parser)]
#[command(
    name = "archgpu",
    version,
    about = "Automate NVIDIA/Wayland/power/gaming setup on Arch Linux",
    long_about = None,
)]
pub struct Cli {
    /// Print detected hardware (chassis, GPU inventory) and bootloader; no changes written
    #[arg(long)]
    pub detect: bool,

    /// Run read-only diagnostic scan and print findings with suggested fixes
    #[arg(long)]
    pub diagnose: bool,

    /// Plan changes but do not write any files or execute privileged commands
    #[arg(long)]
    pub dry_run: bool,

    /// Install Wayland env vars + initramfs module drop-ins for NVIDIA
    #[arg(long)]
    pub apply_wayland: bool,

    /// Ensure `nvidia-drm.modeset=1` is in the kernel command line (UKI)
    #[arg(long)]
    pub apply_bootloader: bool,

    /// Enable nvidia-suspend/hibernate/resume and write modprobe power tweaks
    #[arg(long)]
    pub apply_power: bool,

    /// Enable [multilib], install Vulkan/gamemode/mangohud, and GPU-appropriate drivers
    #[arg(long)]
    pub apply_gaming: bool,

    /// Self-heal pass (Phase 20): deletes legacy PRIME drop-ins on desktop hybrids,
    /// removes `nvidia-prime` where it shouldn't be, forces DKMS rebuild when the
    /// NVIDIA module silently failed to build. Idempotent on healthy hosts.
    #[arg(long)]
    pub apply_repair: bool,

    /// Phase 26: install vendor-agnostic userspace — Vulkan loader, Mesa GL,
    /// split firmware (`linux-firmware-amdgpu` / `linux-firmware-intel`), VA-API
    /// drivers (generation-gated on Intel), and diagnostic tools (vulkan-tools,
    /// clinfo, libva-utils, vdpauinfo). Non-gaming; run before `--apply-gaming`.
    #[arg(long)]
    pub apply_essentials: bool,

    /// Phase 27: provision the invoking user's membership in `video` and `render`
    /// via `usermod -aG video,render <user>`. Requires re-login (not reboot) for
    /// the change to reach the current session.
    #[arg(long)]
    pub apply_groups: bool,

    /// Phase 28: reverse-cleanup pass — removes hardware-absent vendor packages,
    /// defunct (Mesa-2026-bundled) packages, legacy Xorg DDX drivers, AMDVLK
    /// when RADV is also installed. Writes a pre-cleanup snapshot to
    /// /var/backups/archgpu/pre-cleanup-<ts>.txt before removing anything.
    /// Pair with --dry-run to see the plan without removing.
    #[arg(long)]
    pub apply_cleanup: bool,

    /// Phase 29: smart troubleshoot loop. Runs each registered Recipe through
    /// detect → fix → verify. Pair with --dry-run to detect+explain without
    /// writing fixes. Recipes: nomodeset_stuck, nouveau_active_with_nvidia,
    /// dangling_vulkan_icd, software_rendering (diagnostic-only).
    #[arg(long)]
    pub apply_troubleshoot: bool,

    /// Run every apply action (NOTE: does NOT include --apply-cleanup or
    /// --apply-troubleshoot; both are explicit-opt-in only)
    #[arg(long)]
    pub apply_all: bool,

    /// Pass `--noconfirm` to pacman (required for non-interactive / GUI installs)
    #[arg(long, short = 'y')]
    pub yes: bool,

    /// Skip the EUID check (testing only; writes will still fail without privileges)
    #[arg(long, hide = true)]
    pub no_root_check: bool,
}

impl Cli {
    pub fn has_any_action(&self) -> bool {
        self.detect || self.diagnose || self.will_write()
    }

    pub fn will_write(&self) -> bool {
        self.apply_wayland
            || self.apply_bootloader
            || self.apply_power
            || self.apply_gaming
            || self.apply_repair
            || self.apply_essentials
            || self.apply_groups
            || self.apply_cleanup
            || self.apply_troubleshoot
            || self.apply_all
    }

    pub fn needs_root(&self) -> bool {
        if self.no_root_check {
            return false;
        }
        if !self.has_any_action() {
            return true;
        }
        if self.dry_run {
            return false;
        }
        if (self.detect || self.diagnose) && !self.will_write() {
            return false;
        }
        true
    }

    fn actions(&self) -> Actions {
        if self.apply_all {
            // --apply-all may be combined with --apply-cleanup and/or
            // --apply-troubleshoot. all() leaves both off; merge in explicit opt-ins.
            let mut a = Actions::all();
            a.cleanup = self.apply_cleanup;
            a.troubleshoot = self.apply_troubleshoot;
            return a;
        }
        Actions {
            wayland: self.apply_wayland,
            bootloader: self.apply_bootloader,
            power: self.apply_power,
            gaming: self.apply_gaming,
            repair: self.apply_repair,
            essentials: self.apply_essentials,
            groups: self.apply_groups,
            cleanup: self.apply_cleanup,
            troubleshoot: self.apply_troubleshoot,
        }
    }
}

pub fn run(args: Cli) -> Result<()> {
    let mode = if args.dry_run {
        ExecutionMode::DryRun
    } else {
        ExecutionMode::Apply
    };
    let ctx = Context::production(mode);

    let form = hardware::get_chassis_type(&ctx.paths.dmi_chassis).unwrap_or_else(|e| {
        log::warn!("Chassis detection failed: {e:#}. Assuming Unknown form factor.");
        FormFactor::Unknown
    });
    let gpus = GpuInventory::detect().unwrap_or_else(|e| {
        log::warn!("GPU detection failed: {e:#}. Continuing with empty inventory.");
        GpuInventory::default()
    });

    if args.detect {
        print_detection(&ctx, form, &gpus);
    }
    if args.diagnose {
        let findings = diagnostics::scan(&ctx, &gpus, form);
        print_findings(&findings);
    }

    let actions = args.actions();
    if !actions.any() {
        if !args.detect && !args.diagnose {
            log::warn!("No apply-action flags specified. Nothing to do.");
        }
        return Ok(());
    }

    if actions.gaming && !gpus.has_nvidia() {
        if let Some(prim) = gpus.gpus.first() {
            log::info!(
                "No NVIDIA GPU detected; gaming setup will use generic Vulkan + drivers for {:?}.",
                prim.vendor
            );
        }
    }

    log::info!(
        "Executing actions in {} mode: {actions:?}",
        if mode.is_dry_run() {
            "dry-run"
        } else {
            "apply"
        }
    );
    let mut progress = |line: &str| println!("{line}");
    let reports = core::run_actions(&ctx, form, &gpus, actions, args.yes, &mut progress)?;

    println!();
    println!("Results:");
    let mut current: &str = "";
    for (section, r) in &reports {
        if *section != current {
            println!();
            println!("--- {section} ---");
            current = section;
        }
        println!("  {r}");
    }
    println!();

    Ok(())
}

fn print_detection(ctx: &Context, form: FormFactor, gpus: &GpuInventory) {
    println!("ArchGPU — system detection");
    println!("  Form factor          : {form:?}");
    println!(
        "  DMI chassis path     : {}",
        ctx.paths.dmi_chassis.display()
    );
    println!();
    println!("  GPU inventory:");
    if gpus.gpus.is_empty() {
        println!("    (no display controllers detected via lspci)");
    } else {
        for g in &gpus.gpus {
            println!(
                "    - {:?} @ {}  {}  [vendor=0x{:04x} device=0x{:04x}]{}",
                g.vendor,
                g.pci_address,
                g.display_name(),
                g.vendor_id,
                g.device_id,
                if g.is_integrated {
                    "  (integrated)"
                } else {
                    ""
                },
            );
            if let Some(drv) = &g.kernel_driver {
                println!("        kernel driver: {drv}");
            }
            if let Some(gen) = g.nvidia_gen {
                println!(
                    "        NVIDIA arch  : {}  (open kernel modules: {})",
                    gen.human(),
                    if gen.supports_open_modules() {
                        "yes"
                    } else {
                        "no"
                    }
                );
            }
            if let Some(rec) = g.recommended_nvidia_package() {
                let tag = match rec.source {
                    PackageSource::Official => "repo",
                    PackageSource::Aur => "AUR",
                    PackageSource::Unsupported => "unsupported",
                };
                println!(
                    "        recommended  : {} ({tag}) — {}",
                    rec.package, rec.note
                );
            }
        }
    }
    println!();
    println!("  Summary:");
    println!("    NVIDIA present : {}", gpus.has_nvidia());
    println!("    Hybrid (Optimus): {}", gpus.is_hybrid());
    println!();

    let cmdline_exists = ctx.paths.kernel_cmdline.exists();
    println!(
        "  UKI cmdline path     : {} (exists: {cmdline_exists})",
        ctx.paths.kernel_cmdline.display()
    );
    if gpus.has_nvidia() {
        match bootloader::detect(ctx) {
            Ok(mgr) => {
                let present = mgr.has_parameter(NVIDIA_DRM_PARAM).unwrap_or(false);
                println!(
                    "  {NVIDIA_DRM_PARAM} : {}",
                    if present { "present" } else { "missing" }
                );
            }
            Err(e) => println!("  Bootloader detection : failed: {e}"),
        }
    }
    println!(
        "  Pacman config        : {}",
        ctx.paths.pacman_conf.display()
    );
    println!(
        "  Execution mode       : {}",
        if ctx.mode.is_dry_run() {
            "dry-run"
        } else {
            "apply"
        }
    );
    println!();
}

fn print_findings(findings: &[Finding]) {
    println!("ArchGPU — diagnostics");
    if findings.is_empty() {
        println!("  (no findings)");
        return;
    }
    let mut counts = (0usize, 0usize, 0usize);
    for f in findings {
        match f.severity {
            Severity::Info => counts.0 += 1,
            Severity::Warning => counts.1 += 1,
            Severity::Error => counts.2 += 1,
        }
        println!("  {} {}", f.severity.marker(), f.title);
        if !f.detail.is_empty() {
            println!("         {}", f.detail);
        }
        if let Some(fix) = &f.fix_hint {
            println!("    fix: {fix}");
        }
    }
    println!();
    println!(
        "  summary: {} info / {} warnings / {} errors",
        counts.0, counts.1, counts.2
    );
    println!();
}
