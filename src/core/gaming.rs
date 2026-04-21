use anyhow::{Context as _, Result};
use std::collections::HashSet;
use std::process::Command;

use crate::core::aur;
use crate::core::essentials::IntelGenerationCheck;
use crate::core::gpu::{GpuInventory, NvidiaGeneration, PackageSource};
use crate::core::hardware::FormFactor;
use crate::core::state::TweakState;
use crate::core::{Context, ExecutionMode};
use crate::utils::fs_helper::{atomic_write, backup_to_dir, write_dropin, ChangeReport};
use crate::utils::process::run_streaming;

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Phase 18 architectural invariant: NO GLOBAL VULKAN ICD POISONING.
// ═══════════════════════════════════════════════════════════════════════════════════════════
//
// This module must NEVER write a file under `/etc/profile.d/` and must NEVER export
// `VK_DRIVER_FILES`, `VK_ICD_FILENAMES`, `VK_LAYER_PATH`, or `LIBVA_DRIVER_NAME` in any
// system-wide file. Globally pinning the Vulkan loader or VA-API driver blindfolds the
// kernel's DRM-node-based GPU routing — on a hybrid host (e.g. AMD iGPU + NVIDIA dGPU)
// this breaks `prime-run`, Mutter's per-surface offload, and Mesa's automatic vendor
// selection, and was observed to crash GNOME Wayland into llvmpipe software rendering.
//
// The correct abstractions already exist: the kernel exposes one `/dev/dri/renderD*`
// node per GPU; Mesa 23+ and NVIDIA 545+ route per-surface based on DRM lease + GBM;
// per-process offload uses `prime-run` which sets the ICD-selection variables for THAT
// invocation only. Nothing this tool does should override that hierarchy.
//
// Pre-Phase-18 versions of this tool wrote `/etc/profile.d/99-gaming.*` files that set
// ICD variables globally. On every `apply()`, `cleanup_legacy_profile_d` actively
// deletes those artifacts so an upgraded host converges to a clean profile.d state.
//
// These two invariants are enforced by the test suite at the bottom of this file:
//   - `apply_never_creates_a_profile_d_file`
//   - `no_global_icd_sentinels_in_any_written_artifact`
//   - `cleanup_deletes_legacy_99_gaming_sh`
// ═══════════════════════════════════════════════════════════════════════════════════════════

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
    // Phase 25: `goverlay` was briefly in this list (Phase 23 addition) but moved out.
    // It's a convenience GUI for MangoHud config, not structural to the gaming stack —
    // and forcing it into the "required for Active" bar meant existing users with
    // every core package installed saw the gaming tweak stuck as Unapplied forever
    // (the toggle refused to transition to the green ✓ Active badge). Now surfaced
    // as an Info-severity Finding in diagnostics::scan so users who want it can
    // `pacman -S goverlay` themselves.
];

/// Gaming setup is Applied when: `[multilib]` is enabled AND every required repo package
/// for the detected GPU vendor(s) + the always-on set (vulkan-icd-loader / gamemode /
/// mangohud + their lib32 variants) is installed per `pacman -Qq`.
///
/// Never Incompatible — gaming setup is a universal improvement regardless of GPU vendor.
/// `form` is consulted for Phase 19's laptop-vs-desktop hybrid rule (so the package
/// expectation matches what `apply` actually installs).
pub fn check_state(ctx: &Context, gpus: &GpuInventory, form: FormFactor) -> TweakState {
    if !is_multilib_enabled(&ctx.paths.pacman_conf) {
        return TweakState::Unapplied;
    }
    let expected = resolve_gaming_packages(gpus, form);
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
    form: FormFactor,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ChangeReport>> {
    let mut reports = Vec::new();
    // Phase 18: delete legacy `/etc/profile.d/99-gaming.*` artifacts from pre-Phase-18
    // versions of this tool FIRST, so any downstream state probe on an upgraded host
    // sees a clean profile.d. See the architectural invariant at the top of this module.
    reports.extend(cleanup_legacy_profile_d(ctx)?);

    // Phase 19: enable_multilib may modify /etc/pacman.conf. If it does, pacman's
    // in-memory DB has no record of the [multilib] repo until we run `-Sy`, and the
    // subsequent `pacman -S lib32-*` would fail with "target not found". Track the
    // multilib change so we can emit exactly ONE DB-refresh BEFORE the install.
    let multilib_report = enable_multilib(ctx)?;
    let multilib_modified = should_sync_after_multilib(&multilib_report);
    reports.push(multilib_report);

    reports.push(write_sysctl_dropin(ctx)?);

    if multilib_modified {
        reports.push(sync_pacman_db(ctx, progress)?);
    }

    reports.push(install_official_packages(
        ctx, gpus, form, assume_yes, progress,
    )?);

    let aur_pkgs = resolve_aur_packages(gpus);
    if !aur_pkgs.is_empty() {
        let refs: Vec<&str> = aur_pkgs.iter().map(String::as_str).collect();
        reports.push(aur::install_aur_packages(ctx, &refs, assume_yes, progress)?);
    }

    Ok(reports)
}

/// Pure: given the ChangeReport from `enable_multilib`, decide whether pacman needs a
/// `-Sy` database refresh before the next `-S --needed` call.
///
/// Applied  → pacman.conf was freshly modified → YES, must sync to see [multilib].
/// Planned  → dry-run previewed a modification → YES (preview the full flow the user
///            would see in Apply mode; `sync_pacman_db` respects DryRun and returns
///            its own `Planned` rather than spawning pacman).
/// AlreadyApplied → nothing changed → NO (the existing DB either already has multilib or
///                  the user's regular sync cadence applies; not our job to force a refresh).
pub fn should_sync_after_multilib(report: &ChangeReport) -> bool {
    matches!(
        report,
        ChangeReport::Applied { .. } | ChangeReport::Planned { .. }
    )
}

fn sync_pacman_db(ctx: &Context, progress: &mut dyn FnMut(&str)) -> Result<ChangeReport> {
    let detail = "pacman -Sy (refresh DB after enabling [multilib])".to_string();
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned { detail });
    }
    if matches!(ctx.mode, ExecutionMode::Apply) {
        let mut cmd = Command::new("pacman");
        cmd.arg("-Sy");
        progress(&format!("[pacman] {detail}"));
        let status = run_streaming(cmd, |line| progress(&format!("[pacman] {line}")))?;
        if !status.success() {
            anyhow::bail!("pacman -Sy exited with {status}");
        }
    }
    Ok(ChangeReport::Applied {
        detail,
        backup: None,
    })
}

/// Delete any `/etc/profile.d/99-gaming.*` left by pre-Phase-18 archgpu versions.
///
/// Older releases (pre-Phase-18) shipped a profile.d drop-in that exported Vulkan ICD
/// variables globally. On modern hybrid-GPU hosts that file actively breaks video decode
/// routing and Wayland compositor GPU selection. This function removes any such file on
/// every `apply()`, so the tool is self-healing across upgrades.
///
/// Matches files whose stem is EXACTLY `99-gaming` with any extension — e.g.
/// `99-gaming.sh`, `99-gaming.conf`. Does NOT match `99-gaming-user.sh` or `99-gaming`
/// (no extension); those are left alone.
pub fn cleanup_legacy_profile_d(ctx: &Context) -> Result<Vec<ChangeReport>> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(&ctx.paths.profile_d) else {
        // Directory missing is fine — nothing to clean up on a fresh install.
        return Ok(out);
    };
    for entry in rd.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !is_legacy_gaming_profile_d(name) {
            continue;
        }
        let detail = format!(
            "remove legacy ICD-poisoning file {} (pre-Phase-18 archgpu artifact)",
            path.display()
        );
        if ctx.mode.is_dry_run() {
            out.push(ChangeReport::Planned { detail });
            continue;
        }
        let backup = backup_to_dir(&path, &ctx.paths.backup_dir)?;
        std::fs::remove_file(&path)
            .with_context(|| format!("removing legacy profile.d artifact {}", path.display()))?;
        out.push(ChangeReport::Applied { detail, backup });
    }
    Ok(out)
}

fn is_legacy_gaming_profile_d(name: &str) -> bool {
    matches!(name.rsplit_once('.'), Some(("99-gaming", _)))
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
    form: FormFactor,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    let mut packages = resolve_gaming_packages(gpus, form);
    // Phase 18: if EITHER the official queue OR the AUR queue contains any `*-dkms`
    // driver, the build will silently fail without kernel headers. Enumerate every
    // installed kernel package and append the matching `-headers` to the OFFICIAL
    // (pacman) queue — covering users who run linux + linux-lts, linux-zen + linux-lts,
    // etc. Headers always come from the repos, so pacman handles them regardless of
    // whether the driver itself goes through yay. This was the silent failure mode on
    // the RTX 3060 hybrid host: nvidia-open-dkms installed, no linux-headers, module
    // never built, GNOME Wayland fell back to llvmpipe on next login.
    let mut dkms_trigger_set = packages.clone();
    dkms_trigger_set.extend(resolve_aur_packages(gpus));
    let installed_kernels = detect_installed_kernels();
    let headers = kernel_header_packages(&dkms_trigger_set, &installed_kernels);
    for h in headers {
        if !packages.contains(&h) {
            packages.push(h);
        }
    }
    // Phase 25: short-circuit when every package in the install queue is already
    // installed per `pacman -Qq`. pacman -S --needed would silently do nothing in
    // this case, but we'd still spawn it + stream its "up to date — skipping"
    // lines into the log for every package, which is noise. Return a single clean
    // AlreadyApplied report instead, so the UI / CLI reports exactly match the
    // no-op reality and users see "nothing to do" at a glance.
    if matches!(ctx.mode, ExecutionMode::Apply) {
        if let Some(installed) = pacman_query_installed_set(&packages) {
            if packages.iter().all(|p| installed.contains(p.as_str())) {
                return Ok(ChangeReport::AlreadyApplied {
                    detail: format!(
                        "{} package(s) already installed — nothing to do",
                        packages.len()
                    ),
                });
            }
        }
    }

    // Phase 19: use `pacman -S --needed` (NOT `-Syu`) for the gaming install. Rationale:
    // `-u` forces a full system upgrade as a side effect, which is overreach for a
    // targeted "install these gaming packages" action and can surprise users who
    // deliberately pin their pacman upgrade cadence. The DB freshness contract is now
    // owned by `sync_pacman_db` — called earlier in `apply()` exactly when
    // `enable_multilib` modified pacman.conf.
    let detail = format!("pacman -S --needed {}", packages.join(" "));

    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned { detail });
    }
    if matches!(ctx.mode, ExecutionMode::Apply) {
        let mut cmd = Command::new("pacman");
        cmd.arg("-S").arg("--needed");
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
///
/// Phase 19: `form` drives the Optimus/PRIME gate — `nvidia-prime` (the `prime-run`
/// wrapper) is only added when both the GPU inventory is hybrid AND the chassis is a
/// Laptop. On a desktop tower with NVIDIA + iGPU, the physical display cable dictates
/// the primary GPU and render-offload tooling is a no-op at best, a misconfiguration
/// at worst. Unknown form factors are treated as non-laptop (conservative: a false
/// negative leaves the user without `prime-run`, easily recovered; a false positive
/// on a desktop can break display routing).
pub fn resolve_gaming_packages(gpus: &GpuInventory, form: FormFactor) -> Vec<String> {
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
        // Phase 19: PRIME is a LAPTOP concept. On a hybrid desktop (NVIDIA + iGPU in a
        // tower), `nvidia-prime` provides `prime-run` which sets __NV_PRIME_RENDER_OFFLOAD=1
        // — useless on a desktop where the monitor is plugged into one specific GPU and
        // the kernel already routes display via the DRM node on that card.
        if gpus.is_hybrid() && form == FormFactor::Laptop {
            add("nvidia-prime");
        }
    }
    if gpus.has_amd() {
        // Phase 23: the 32-bit OpenGL runtime is required for classic Steam/Proton
        // titles that don't use Vulkan. The Arch Wiki AMDGPU install page
        // explicitly calls for `mesa` + `lib32-mesa`. Pre-Phase-23 we installed
        // lib32-vulkan-radeon (Vulkan userspace) but NOT lib32-mesa (GL
        // userspace), breaking older Steam library titles. `mesa` is usually
        // already present via base deps but explicit install is idempotent under
        // `pacman -S --needed`.
        add("mesa");
        add("lib32-mesa");
        // RADV (vulkan-radeon) is Mesa's Vulkan driver — faster and more widely
        // tested in Proton than AMD's own AMDVLK. Sanitation code warns if the
        // user has amdvlk/lib32-amdvlk installed alongside.
        add("vulkan-radeon");
        add("lib32-vulkan-radeon");
        // Phase 26: `libva-mesa-driver` + `lib32-libva-mesa-driver` were removed —
        // those packages no longer exist on Arch. Mesa bundles its own VA-API
        // backend as of the 2025 Gallium consolidation, so `mesa` / `lib32-mesa`
        // above are sufficient. Earlier versions of this tool tried to install
        // the now-defunct names and hard-failed on any fresh system.
    }
    if gpus.has_intel() {
        // Phase 23: same 32-bit-OpenGL-runtime rationale as AMD above. Intel
        // users hit the same broken-Steam-titles bug on pre-Phase-23 tool runs.
        add("mesa");
        add("lib32-mesa");
        add("vulkan-intel");
        add("lib32-vulkan-intel");
        // Phase 26: generation-gate the VA-API driver instead of unconditionally
        // installing `intel-media-driver` (iHD). iHD covers Broadwell (Gen 8) and
        // newer; Gen 6/7 (Sandy Bridge / Ivy Bridge / Haswell) need the legacy
        // `libva-intel-driver` (i965). Boundary heuristic lives in
        // `core::essentials`.
        if gpus.any_pre_broadwell_intel() {
            add("libva-intel-driver");
        } else {
            add("intel-media-driver");
        }
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

    // Phase 23: AMD DDX analogue — `xf86-video-amdgpu` is the old DDX-era Xorg driver
    // for amdgpu-class cards. The Arch Wiki AMDGPU page explicitly recommends the
    // generic `modesetting` driver (shipped in xorg-server) as the default, with DDX
    // only needed for TearFree or pre-GCN edge cases. DDX has no Wayland story and is
    // known to cause tearing on recent Mesa/kernel combinations.
    if gpus.has_amd() && installed.contains("xf86-video-amdgpu") {
        out.push(SanitationWarning {
            title: "Legacy xf86-video-amdgpu DDX installed".into(),
            detail: "`xf86-video-amdgpu` is the old DDX-era Xorg driver. The modesetting driver \
                     (shipped in xorg-server) is the modern replacement for amdgpu-class cards \
                     on both Xorg and Wayland. Unless you specifically need TearFree on a \
                     pre-GCN card, you don't want this package installed."
                .into(),
            remediation: "sudo pacman -Rns xf86-video-amdgpu".into(),
        });
    }

    // Mesa 25 legacy — VDPAU for the Gallium drivers (AMD / Intel / Nouveau) was
    // removed in Mesa 25. The `mesa-vdpau` / `libva-mesa-driver` packages no longer
    // exist on Arch at all (Mesa 26 bundles VA-API in-tree and VDPAU is gone);
    // `vdpauinfo` stays useful for NVIDIA's VDPAU backend only.
    let defunct_found: Vec<&str> = [
        "mesa-vdpau",
        "lib32-mesa-vdpau",
        "libva-mesa-driver",
        "lib32-libva-mesa-driver",
    ]
    .into_iter()
    .filter(|n| installed.contains(*n))
    .collect();
    if !defunct_found.is_empty() {
        out.push(SanitationWarning {
            title: "Legacy Mesa VA-API / VDPAU packages installed".into(),
            detail: format!(
                "Found: {}. Mesa 26 bundles VA-API (and dropped VDPAU) in-tree — these split \
                 packages no longer exist on Arch and linger only from pre-2026 upgrades.",
                defunct_found.join(", ")
            ),
            remediation: format!("sudo pacman -Rns {}", defunct_found.join(" ")),
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
        "xf86-video-amdgpu",
        "mesa-vdpau",
        "lib32-mesa-vdpau",
        "libva-mesa-driver",
        "lib32-libva-mesa-driver",
    ];
    let owned: Vec<String> = candidates.iter().map(|s| s.to_string()).collect();
    let installed = pacman_query_installed_set(&owned).unwrap_or_default();
    sanitation_warnings_from_installed(gpus, &installed)
}

// ── Phase 18: dynamic kernel header mapping ─────────────────────────────────────────────────
//
// Every official Arch kernel package ships a matching `<pkg>-headers` package that exports
// the build tree DKMS needs to compile `nvidia-open-dkms` / `nvidia-dkms` / `nvidia-470xx-dkms`
// / `nvidia-390xx-dkms` against. Install the driver without headers and DKMS silently
// reports success while producing no kernel module — the GPU comes up driverless on next
// boot. This was the original Phase 18 bug report on the RTX 3060 hybrid.
//
// The mapping is 1:1 — for every `linux*` base package the user has installed, install
// `linux*-headers`. We do NOT use `uname -r` because a user running `linux-lts` today may
// also have `linux` installed (and vice-versa); mapping only the RUNNING kernel would still
// leave the other kernel's DKMS build broken on the next boot into it.

/// Every Arch-official kernel base package we know how to map to a `-headers` counterpart.
/// Source: <https://archlinux.org/packages/?q=linux&repo=core>.
const KNOWN_KERNEL_PACKAGES: &[&str] = &[
    "linux",
    "linux-lts",
    "linux-zen",
    "linux-hardened",
    "linux-rt",
    "linux-rt-lts",
];

/// Pure: from the raw output of `pacman -Qq`, return the kernel base packages we recognize
/// (NOT their `-headers` or `-docs` variants). Order-preserving by input order.
///
/// Deliberately uses a whitelist (`KNOWN_KERNEL_PACKAGES`) rather than a regex like
/// `^linux(-[a-z]+)?$`, because that regex would match `linux-api-headers`, `linux-firmware`,
/// `linux-tools`, and any out-of-tree `linux-<anything>` meta-package. False positives here
/// translate to `pacman -S linux-firmware-headers` — which doesn't exist and fails the
/// whole install batch.
pub fn parse_installed_kernels(pacman_qq: &str) -> Vec<String> {
    let known: HashSet<&str> = KNOWN_KERNEL_PACKAGES.iter().copied().collect();
    pacman_qq
        .lines()
        .map(str::trim)
        .filter(|l| known.contains(l))
        .map(ToOwned::to_owned)
        .collect()
}

/// Runtime probe: query the local pacman DB for installed kernels. Falls back to
/// `vec!["linux"]` if pacman is unavailable (test sandbox, container) or reports no
/// recognized kernels — installing `linux-headers` on a host with no base `linux` is a
/// no-op (`--needed` skips it), whereas skipping headers entirely reproduces the bug.
pub fn detect_installed_kernels() -> Vec<String> {
    let Ok(out) = Command::new("pacman").arg("-Qq").output() else {
        return vec!["linux".to_string()];
    };
    if !out.status.success() {
        return vec!["linux".to_string()];
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let kernels = parse_installed_kernels(&body);
    if kernels.is_empty() {
        vec!["linux".to_string()]
    } else {
        kernels
    }
}

/// Pure: given a prospective install queue and the set of installed kernels, return the
/// `-headers` packages that need to be appended to the queue. Empty when no DKMS package
/// is queued (so ordinary RADV / vulkan-intel-only hosts don't pull in headers they don't
/// need).
///
/// Any package ending in `-dkms` triggers the mapping — this covers `nvidia-open-dkms`,
/// `nvidia-580xx-dkms`, `nvidia-470xx-dkms`, `nvidia-390xx-dkms`, and any future DKMS
/// driver added to `resolve_gaming_packages` or `resolve_aur_packages`.
pub fn kernel_header_packages(
    install_queue: &[String],
    installed_kernels: &[String],
) -> Vec<String> {
    let needs_headers = install_queue.iter().any(|p| p.ends_with("-dkms"));
    if !needs_headers {
        return Vec::new();
    }
    installed_kernels
        .iter()
        .map(|k| format!("{k}-headers"))
        .collect()
}

/// AUR packages the tool will build+install via yay. Empty for Turing+ hosts.
pub fn resolve_aur_packages(gpus: &GpuInventory) -> Vec<String> {
    let Some(nv) = gpus.primary_nvidia() else {
        return Vec::new();
    };
    let gen = nv.nvidia_gen.unwrap_or(NvidiaGeneration::Unknown);

    match gen {
        // Phase 24 correction: Maxwell / Volta / Pascal share the 580.x AUR branch.
        // Previously Maxwell was routed to 470xx and Pascal/Volta went to the
        // (now non-existent) repo nvidia-dkms package. See gpu.rs for the narrative.
        NvidiaGeneration::Maxwell
        | NvidiaGeneration::Volta
        | NvidiaGeneration::Pascal => vec![
            "nvidia-580xx-dkms".into(),
            "nvidia-580xx-utils".into(),
            "lib32-nvidia-580xx-utils".into(),
        ],
        NvidiaGeneration::Kepler => vec![
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
        let pkgs = resolve_gaming_packages(&inv(vec![intel(0x64a0)]), FormFactor::Laptop);
        assert!(!pkgs.iter().any(|p| p.starts_with("nvidia")));
        assert!(pkgs.contains(&"vulkan-intel".to_string()));
        // Phase 15: Intel VA-API via intel-media-driver.
        assert!(pkgs.contains(&"intel-media-driver".to_string()));
        // Phase 23: 32-bit OpenGL runtime for Intel — required for classic Steam titles.
        assert!(pkgs.contains(&"mesa".to_string()));
        assert!(pkgs.contains(&"lib32-mesa".to_string()));
    }

    #[test]
    fn amd_gets_mesa_and_radv_without_defunct_libva_mesa_driver() {
        let pkgs = resolve_gaming_packages(&inv(vec![amd(0x73bf)]), FormFactor::Desktop);
        // Phase 26: `libva-mesa-driver` + `lib32-libva-mesa-driver` no longer exist
        // on Arch (Mesa bundles its own VA-API backend). They must NOT appear in the
        // install list — doing so would hard-fail `pacman -S` on every fresh install.
        assert!(!pkgs.contains(&"libva-mesa-driver".to_string()));
        assert!(!pkgs.contains(&"lib32-libva-mesa-driver".to_string()));
        assert!(pkgs.contains(&"vulkan-radeon".to_string()));
        assert!(pkgs.contains(&"lib32-vulkan-radeon".to_string()));
        assert!(
            !pkgs.iter().any(|p| p == "amdvlk"),
            "never recommend AMDVLK"
        );
        // Phase 23: 32-bit OpenGL runtime for AMD — Arch Wiki AMDGPU install page
        // explicitly calls for this. Without lib32-mesa, classic (non-Vulkan) Steam
        // titles fall back to llvmpipe for 32-bit GL calls.
        assert!(pkgs.contains(&"mesa".to_string()));
        assert!(pkgs.contains(&"lib32-mesa".to_string()));
    }

    #[test]
    fn intel_pre_broadwell_host_gets_legacy_i965_va_driver() {
        // Phase 26: Gen 6/7 Intel (Sandy Bridge / Ivy Bridge / Haswell — device ID
        // < 0x1600) gets `libva-intel-driver` instead of `intel-media-driver`.
        // iHD doesn't support these older gens.
        let pkgs = resolve_gaming_packages(&inv(vec![intel(0x0166)]), FormFactor::Laptop);
        assert!(pkgs.contains(&"libva-intel-driver".to_string()));
        assert!(!pkgs.contains(&"intel-media-driver".to_string()));
    }

    #[test]
    fn nvidia_only_does_not_get_mesa_in_explicit_install_list() {
        // Phase 23 invariant: NVIDIA-only hosts don't need `mesa` / `lib32-mesa` in the
        // explicit install list because nvidia-utils provides its own GL stack and
        // libglvnd dispatches correctly. Mesa may still be present transitively via
        // `vulkan-icd-loader` optdepends — that's fine and out of this test's scope.
        let pkgs = resolve_gaming_packages(
            &inv(vec![nvidia(NvidiaGeneration::Ada, 0x2684)]),
            FormFactor::Desktop,
        );
        assert!(
            !pkgs.iter().any(|p| p == "mesa" || p == "lib32-mesa"),
            "NVIDIA-only host should not have mesa explicitly installed: {pkgs:?}"
        );
    }

    #[test]
    fn always_on_set_excludes_goverlay() {
        // Phase 25: goverlay moved OUT of the always-on list. It was briefly required
        // in Phase 23 but forcing a cosmetic tool into the Active-state bar left
        // existing users stuck — the gaming tweak couldn't transition to ✓ Active
        // without them running Apply again just to pull in goverlay. Now diagnostics
        // surfaces it as an Info Finding instead.
        let pkgs = resolve_gaming_packages(&inv(vec![intel(0x64a0)]), FormFactor::Laptop);
        assert!(
            !pkgs.contains(&"goverlay".to_string()),
            "goverlay must not be in the required install list anymore: {pkgs:?}"
        );
    }

    #[test]
    fn nvidia_gets_libva_nvidia_driver() {
        let pkgs = resolve_gaming_packages(
            &inv(vec![nvidia(NvidiaGeneration::Ada, 0x2684)]),
            FormFactor::Desktop,
        );
        // Phase 15: NVIDIA VA-API via libva-nvidia-driver.
        assert!(pkgs.contains(&"libva-nvidia-driver".to_string()));
    }

    #[test]
    fn nvidia_ada_desktop_gets_open_dkms_but_not_prime() {
        let pkgs = resolve_gaming_packages(
            &inv(vec![nvidia(NvidiaGeneration::Ada, 0x2684)]),
            FormFactor::Desktop,
        );
        assert!(pkgs.contains(&"nvidia-open-dkms".to_string()));
        assert!(!pkgs.iter().any(|p| p == "nvidia-prime"));
    }

    #[test]
    fn laptop_hybrid_gets_prime() {
        // Laptop + hybrid → prime-run is meaningful (Optimus offload). Regression guard:
        // if the form-factor gate ever starts returning the wrong branch on laptops,
        // this test catches it.
        let pkgs = resolve_gaming_packages(
            &inv(vec![
                intel(0x3e9b),
                nvidia(NvidiaGeneration::Ampere, 0x25a2),
            ]),
            FormFactor::Laptop,
        );
        assert!(pkgs.contains(&"nvidia-prime".to_string()));
    }

    #[test]
    fn desktop_hybrid_does_not_get_prime() {
        // Phase 19 core guarantee: the RTX 3060 + AMD iGPU field-test topology. Hybrid,
        // but on a desktop — physical cable dictates display, prime-run is useless and
        // its OutputClass drop-in would actively misroute. nvidia-prime must NOT appear.
        let pkgs = resolve_gaming_packages(
            &inv(vec![
                amd(0x1638),
                nvidia(NvidiaGeneration::Ampere, 0x2504),
            ]),
            FormFactor::Desktop,
        );
        assert!(
            !pkgs.iter().any(|p| p == "nvidia-prime"),
            "Phase 19 invariant: desktop hybrid must not install nvidia-prime, got: {pkgs:?}"
        );
        // Everything else NVIDIA should still be there.
        assert!(pkgs.contains(&"nvidia-utils".to_string()));
        assert!(pkgs.contains(&"libva-nvidia-driver".to_string()));
    }

    #[test]
    fn unknown_form_factor_hybrid_does_not_get_prime() {
        // Chassis detection failed. Conservative default: treat as non-laptop to avoid
        // misrouting a desktop's display. User can install nvidia-prime manually if they
        // know they're on a laptop with failed SMBIOS.
        let pkgs = resolve_gaming_packages(
            &inv(vec![
                intel(0x3e9b),
                nvidia(NvidiaGeneration::Ampere, 0x25a2),
            ]),
            FormFactor::Unknown,
        );
        assert!(!pkgs.iter().any(|p| p == "nvidia-prime"));
    }

    #[test]
    fn pascal_driver_is_aur_580xx_not_official() {
        // Phase 24 invariant: Pascal's driver ships via `nvidia-580xx-dkms` in AUR.
        // Pre-Phase-24 we routed Pascal to `nvidia-dkms` in the official repo — a
        // package that no longer exists (extra only has nvidia-open-dkms), causing
        // install failure on every Pascal host.
        let pkgs = resolve_gaming_packages(
            &inv(vec![nvidia(NvidiaGeneration::Pascal, 0x1B06)]),
            FormFactor::Desktop,
        );
        // Neither the ghost `nvidia-dkms` nor `nvidia-open-dkms` (open modules need
        // GSP firmware = Turing+) should appear in the official list.
        assert!(
            !pkgs.iter().any(|p| p == "nvidia-dkms"),
            "Pascal must NOT route to the defunct nvidia-dkms: {pkgs:?}"
        );
        assert!(!pkgs.iter().any(|p| p == "nvidia-open-dkms"));
        // And the AUR list handles the real install.
        let aur = resolve_aur_packages(&inv(vec![nvidia(NvidiaGeneration::Pascal, 0x1B06)]));
        assert!(aur.iter().any(|p| p == "nvidia-580xx-dkms"));
    }

    #[test]
    fn maxwell_official_list_excludes_aur_driver() {
        let pkgs = resolve_gaming_packages(
            &inv(vec![nvidia(NvidiaGeneration::Maxwell, 0x13C2)]),
            FormFactor::Desktop,
        );
        // Phase 24: Maxwell is now on the 580xx branch (not 470xx). Official list
        // must not contain either the old or new AUR driver.
        assert!(!pkgs.iter().any(|p| p.contains("470xx")));
        assert!(!pkgs.iter().any(|p| p.contains("580xx")));
    }

    #[test]
    fn maxwell_volta_pascal_share_580xx_aur_branch() {
        // Phase 24 correction: Maxwell, Volta, and Pascal all share the 580.x legacy
        // driver branch via `nvidia-580xx-dkms` (AUR, maintained by ventureo/CachyOS).
        // Pre-Phase-24 Maxwell was routed to 470xx (Kepler's branch) and Pascal/Volta
        // went to the non-existent official `nvidia-dkms` — both broken.
        for gen in [
            NvidiaGeneration::Maxwell,
            NvidiaGeneration::Volta,
            NvidiaGeneration::Pascal,
        ] {
            let aur = resolve_aur_packages(&inv(vec![nvidia(gen, 0x13C2)]));
            assert!(
                aur.iter().any(|p| p == "nvidia-580xx-dkms"),
                "{gen:?} must route to nvidia-580xx-dkms, got: {aur:?}"
            );
            assert!(aur.iter().any(|p| p == "nvidia-580xx-utils"));
            assert!(aur.iter().any(|p| p == "lib32-nvidia-580xx-utils"));
        }
    }

    #[test]
    fn kepler_stays_on_470xx_branch() {
        // Kepler is a proper subset of what 470xx supports and doesn't move to 580.
        let aur = resolve_aur_packages(&inv(vec![nvidia(NvidiaGeneration::Kepler, 0x100C)]));
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

    // Phase 23: xf86-video-amdgpu sanitation analogue. The modesetting driver is the
    // wiki-recommended default for amdgpu-class cards on both Xorg and Wayland.
    #[test]
    fn sanitation_flags_xf86_video_amdgpu_on_amd_host() {
        let w = sanitation_warnings_from_installed(
            &inv(vec![amd(0x73bf)]),
            &installed_set(&["xf86-video-amdgpu"]),
        );
        assert!(w.iter().any(|x| x.title.contains("xf86-video-amdgpu")));
    }

    #[test]
    fn sanitation_ignores_xf86_video_amdgpu_on_non_amd_host() {
        // Unusual but possible: user has the package installed on an Intel-only host.
        // Not our concern — the warning is specifically "you have AMD and this legacy
        // package on top of it"; unrelated hosts don't get noise.
        let w = sanitation_warnings_from_installed(
            &inv(vec![intel(0x64a0)]),
            &installed_set(&["xf86-video-amdgpu"]),
        );
        assert!(w.iter().all(|x| !x.title.contains("xf86-video-amdgpu")));
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

    // ── Phase 18: ICD guard + legacy profile.d cleanup ────────────────────────────────────

    use tempfile::tempdir;

    #[test]
    fn cleanup_deletes_legacy_99_gaming_sh() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();
        let legacy = ctx.paths.profile_d.join("99-gaming.sh");
        std::fs::write(
            &legacy,
            "export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json\n",
        )
        .unwrap();

        let reports = cleanup_legacy_profile_d(&ctx).unwrap();

        assert!(!legacy.exists(), "legacy 99-gaming.sh must be deleted");
        assert_eq!(reports.len(), 1);
        assert!(matches!(reports[0], ChangeReport::Applied { .. }));
    }

    #[test]
    fn cleanup_deletes_legacy_99_gaming_conf() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();
        let legacy = ctx.paths.profile_d.join("99-gaming.conf");
        std::fs::write(&legacy, "# old archgpu ICD drop-in\n").unwrap();

        cleanup_legacy_profile_d(&ctx).unwrap();

        assert!(!legacy.exists());
    }

    #[test]
    fn cleanup_backs_up_before_deleting() {
        // Deletion is destructive. Verify the pre-delete contents land in backup_dir so
        // a panicked admin can recover.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();
        let legacy = ctx.paths.profile_d.join("99-gaming.sh");
        std::fs::write(&legacy, "LEGACY-MARKER\n").unwrap();

        let reports = cleanup_legacy_profile_d(&ctx).unwrap();

        let ChangeReport::Applied { backup, .. } = &reports[0] else {
            panic!("expected Applied, got {:?}", reports[0]);
        };
        let backup = backup.as_ref().expect("backup path must be present");
        assert!(backup.exists());
        assert_eq!(std::fs::read_to_string(backup).unwrap(), "LEGACY-MARKER\n");
    }

    #[test]
    fn cleanup_spares_unrelated_profile_d_files() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();

        // User's own script — must NOT be touched.
        let user_script = ctx.paths.profile_d.join("99-user.sh");
        std::fs::write(&user_script, "export MY_THING=1\n").unwrap();

        // Prefix-only hyphen-extension match — also must NOT be touched (not a stem match).
        let hyphenated = ctx.paths.profile_d.join("99-gaming-custom.sh");
        std::fs::write(&hyphenated, "export OTHER=1\n").unwrap();

        // Our OWN modernized drop-in (wayland) — different prefix, must NOT be touched.
        let wayland_dropin = ctx.paths.profile_d.join("99-nvidia-wayland.sh");
        std::fs::write(&wayland_dropin, "# archgpu wayland drop-in\n").unwrap();

        cleanup_legacy_profile_d(&ctx).unwrap();

        assert!(user_script.exists());
        assert!(hyphenated.exists());
        assert!(wayland_dropin.exists());
    }

    #[test]
    fn cleanup_is_noop_when_profile_d_missing() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        // profile_d intentionally NOT created — simulates a fresh install.
        let reports = cleanup_legacy_profile_d(&ctx).unwrap();
        assert!(reports.is_empty());
    }

    #[test]
    fn cleanup_dry_run_reports_but_does_not_delete() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();
        let legacy = ctx.paths.profile_d.join("99-gaming.sh");
        std::fs::write(&legacy, "export VK_ICD_FILENAMES=/tmp/x.json\n").unwrap();

        let reports = cleanup_legacy_profile_d(&ctx).unwrap();

        assert_eq!(reports.len(), 1);
        assert!(matches!(reports[0], ChangeReport::Planned { .. }));
        assert!(legacy.exists(), "dry-run must never delete files");
    }

    #[test]
    fn is_legacy_stem_matcher_is_strict() {
        // Matches only files whose stem is EXACTLY `99-gaming`.
        assert!(is_legacy_gaming_profile_d("99-gaming.sh"));
        assert!(is_legacy_gaming_profile_d("99-gaming.conf"));
        assert!(is_legacy_gaming_profile_d("99-gaming.bash"));
        // Doesn't match near-misses.
        assert!(!is_legacy_gaming_profile_d("99-gaming"));
        assert!(!is_legacy_gaming_profile_d("99-gaming-custom.sh"));
        assert!(!is_legacy_gaming_profile_d("98-gaming.sh"));
        assert!(!is_legacy_gaming_profile_d("99-nvidia-wayland.sh"));
    }

    #[test]
    fn no_global_icd_sentinels_in_any_written_artifact() {
        // Architectural invariant test: scan every string CONSTANT this module writes to
        // disk and the actual rendered content of write_sysctl_dropin, for any
        // ICD-poisoning variable. If a future change starts emitting these, this test
        // fails loudly — preventing the Phase 18 bug from silently regressing.
        const ICD_SENTINELS: &[&str] = &[
            "VK_DRIVER_FILES",
            "VK_ICD_FILENAMES",
            "VK_LAYER_PATH",
            "LIBVA_DRIVER_NAME",
        ];

        // The one string constant we own.
        for sentinel in ICD_SENTINELS {
            assert!(
                !SYSCTL_CONTENT.contains(sentinel),
                "SYSCTL_CONTENT regressed and contains {sentinel}"
            );
        }

        // Render the sysctl drop-in and scan its on-disk bytes.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        std::fs::create_dir_all(&ctx.paths.sysctl_d).unwrap();
        write_sysctl_dropin(&ctx).unwrap();
        let body = std::fs::read_to_string(ctx.paths.sysctl_d.join(SYSCTL_DROPIN_FILE)).unwrap();
        for sentinel in ICD_SENTINELS {
            assert!(
                !body.contains(sentinel),
                "sysctl drop-in regressed and contains {sentinel}"
            );
        }
    }

    #[test]
    fn apply_never_creates_a_profile_d_file() {
        // Architectural invariant: regardless of GPU vendor, gaming::apply must not drop
        // ANYTHING into /etc/profile.d/. In dry-run (no pacman invocation) the directory
        // must end exactly as empty as it started.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();
        std::fs::create_dir_all(&ctx.paths.sysctl_d).unwrap();
        // Minimal pacman.conf so enable_multilib doesn't error out.
        std::fs::create_dir_all(ctx.paths.pacman_conf.parent().unwrap()).unwrap();
        std::fs::write(&ctx.paths.pacman_conf, SAMPLE_ENABLED).unwrap();

        for (gpus, form) in [
            (inv(vec![amd(0x73bf)]), FormFactor::Desktop),
            (inv(vec![intel(0x64a0)]), FormFactor::Laptop),
            (
                inv(vec![nvidia(NvidiaGeneration::Ada, 0x2684)]),
                FormFactor::Desktop,
            ),
            (
                inv(vec![intel(0x3e9b), nvidia(NvidiaGeneration::Ampere, 0x25a2)]),
                FormFactor::Laptop,
            ),
        ] {
            apply(&ctx, &gpus, form, false, &mut |_| {}).unwrap();
            let count = std::fs::read_dir(&ctx.paths.profile_d).unwrap().count();
            assert_eq!(
                count, 0,
                "gaming::apply wrote something to /etc/profile.d/ — Phase 18 invariant violated"
            );
        }
    }

    // ── Phase 18: dynamic kernel header mapping ───────────────────────────────────────────

    #[test]
    fn parse_installed_kernels_detects_stock_and_variants() {
        let pacman_qq = "\
base
base-devel
linux
linux-lts
linux-zen
linux-headers
linux-hardened
linux-rt
linux-rt-lts
nvidia-utils
vulkan-icd-loader
";
        let kernels = parse_installed_kernels(pacman_qq);
        assert_eq!(
            kernels,
            vec![
                "linux".to_string(),
                "linux-lts".to_string(),
                "linux-zen".to_string(),
                "linux-hardened".to_string(),
                "linux-rt".to_string(),
                "linux-rt-lts".to_string()
            ]
        );
    }

    #[test]
    fn parse_installed_kernels_excludes_headers_docs_firmware() {
        // Regression guard: the whitelist must not accidentally include -headers, -docs,
        // -firmware, -api-headers, linux-tools, etc. Those are NOT kernel packages and
        // mapping them to `<pkg>-headers` would produce nonexistent names like
        // `linux-firmware-headers`, failing the whole pacman batch.
        let pacman_qq = "\
linux-headers
linux-api-headers
linux-firmware
linux-firmware-whence
linux-docs
linux-tools
linux-tkg-bmq
";
        assert!(parse_installed_kernels(pacman_qq).is_empty());
    }

    #[test]
    fn parse_installed_kernels_on_empty_input_returns_empty() {
        assert!(parse_installed_kernels("").is_empty());
    }

    #[test]
    fn kernel_header_mapping_stock_kernel() {
        let queue = vec![
            "nvidia-open-dkms".to_string(),
            "vulkan-icd-loader".to_string(),
        ];
        let kernels = vec!["linux".to_string()];
        let headers = kernel_header_packages(&queue, &kernels);
        assert_eq!(headers, vec!["linux-headers".to_string()]);
    }

    #[test]
    fn kernel_header_mapping_multi_kernel_host() {
        // RTX 3060 hybrid host scenario — user runs `linux` as primary and `linux-lts`
        // as a fallback. Both need headers so whichever one they boot into, DKMS works.
        let queue = vec!["nvidia-dkms".to_string()];
        let kernels = vec![
            "linux".to_string(),
            "linux-lts".to_string(),
            "linux-zen".to_string(),
        ];
        let headers = kernel_header_packages(&queue, &kernels);
        assert_eq!(
            headers,
            vec![
                "linux-headers".to_string(),
                "linux-lts-headers".to_string(),
                "linux-zen-headers".to_string()
            ]
        );
    }

    #[test]
    fn kernel_header_mapping_skipped_when_no_dkms_queued() {
        // AMD-only host — RADV is a userspace Mesa driver, no kernel module to build,
        // no headers required. Never bloat the install with unnecessary packages.
        let queue = vec![
            "vulkan-radeon".to_string(),
            "lib32-vulkan-radeon".to_string(),
            "gamemode".to_string(),
        ];
        let kernels = vec!["linux".to_string(), "linux-zen".to_string()];
        let headers = kernel_header_packages(&queue, &kernels);
        assert!(headers.is_empty());
    }

    #[test]
    fn kernel_header_mapping_triggered_by_any_dkms_suffix() {
        // The `-dkms` suffix check must catch every driver variant: nvidia-open-dkms,
        // nvidia-580xx-dkms, nvidia-470xx-dkms, nvidia-390xx-dkms.
        let kernels = vec!["linux".to_string()];
        for driver in [
            "nvidia-open-dkms",
            "nvidia-580xx-dkms",
            "nvidia-470xx-dkms",
            "nvidia-390xx-dkms",
        ] {
            let queue = vec![driver.to_string()];
            assert_eq!(
                kernel_header_packages(&queue, &kernels),
                vec!["linux-headers".to_string()],
                "driver {driver} failed to trigger header install"
            );
        }
    }

    #[test]
    fn detect_installed_kernels_falls_back_to_linux_when_pacman_absent() {
        // We can't remove pacman from the test host, but we can verify the contract:
        // the function always returns at least one kernel so the downstream DKMS fix is
        // never silently skipped. On any real Arch host this returns the real set;
        // elsewhere (CI without pacman) it returns ["linux"].
        let kernels = detect_installed_kernels();
        assert!(!kernels.is_empty(), "detect_installed_kernels must never return empty");
    }

    // ── Phase 19: multilib auto-sync ──────────────────────────────────────────────────────

    #[test]
    fn should_sync_after_multilib_fires_on_applied_and_planned_only() {
        // Apply-mode modification → real sync fires.
        assert!(should_sync_after_multilib(&ChangeReport::Applied {
            detail: "uncommented".into(),
            backup: None,
        }));
        // Dry-run modification → planned sync appears in the preview (so the user sees
        // the full sequence). `sync_pacman_db` itself honors DryRun and emits Planned
        // rather than spawning pacman — no real DB mutation occurs.
        assert!(should_sync_after_multilib(&ChangeReport::Planned {
            detail: "would uncomment".into(),
        }));
        // Nothing changed → don't force a refresh.
        assert!(!should_sync_after_multilib(&ChangeReport::AlreadyApplied {
            detail: "already on".into(),
        }));
    }

    #[test]
    fn apply_plans_a_sync_after_multilib_modification_in_dry_run() {
        // Dry-run integration check: on a tempdir whose pacman.conf has multilib commented
        // out, apply()'s reports must include a `Planned` entry describing `pacman -Sy` —
        // proving the sync step is wired in BEFORE install_official_packages.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();
        std::fs::create_dir_all(&ctx.paths.sysctl_d).unwrap();
        std::fs::create_dir_all(ctx.paths.pacman_conf.parent().unwrap()).unwrap();
        std::fs::write(&ctx.paths.pacman_conf, SAMPLE_COMMENTED).unwrap();

        let reports = apply(
            &ctx,
            &inv(vec![amd(0x73bf)]),
            FormFactor::Desktop,
            false,
            &mut |_| {},
        )
        .unwrap();

        let has_sync_plan = reports.iter().any(|r| match r {
            ChangeReport::Planned { detail } => detail.contains("pacman -Sy"),
            _ => false,
        });
        assert!(
            has_sync_plan,
            "apply() must plan a pacman -Sy when multilib was freshly uncommented: reports={reports:#?}"
        );
    }

    #[test]
    fn apply_does_not_sync_when_multilib_already_enabled_dry_run() {
        // Inverse: multilib already on → no sync needed, no sync plan emitted.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        std::fs::create_dir_all(&ctx.paths.profile_d).unwrap();
        std::fs::create_dir_all(&ctx.paths.sysctl_d).unwrap();
        std::fs::create_dir_all(ctx.paths.pacman_conf.parent().unwrap()).unwrap();
        std::fs::write(&ctx.paths.pacman_conf, SAMPLE_ENABLED).unwrap();

        let reports = apply(
            &ctx,
            &inv(vec![amd(0x73bf)]),
            FormFactor::Desktop,
            false,
            &mut |_| {},
        )
        .unwrap();

        let has_sync_plan = reports.iter().any(|r| match r {
            ChangeReport::Planned { detail } => detail.contains("pacman -Sy"),
            _ => false,
        });
        assert!(
            !has_sync_plan,
            "apply() must skip pacman -Sy when multilib was already enabled: reports={reports:#?}"
        );
    }

    #[test]
    fn install_command_no_longer_forces_system_upgrade() {
        // Phase 19: the install command dropped `-u` (no forced system upgrade). Verify the
        // rendered detail string reflects `-S --needed`, not `-Syu --needed`.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        let report = install_official_packages(
            &ctx,
            &inv(vec![amd(0x73bf)]),
            FormFactor::Desktop,
            false,
            &mut |_| {},
        )
        .unwrap();
        let detail = match report {
            ChangeReport::Planned { detail } => detail,
            other => panic!("expected Planned under DryRun, got {other:?}"),
        };
        assert!(
            detail.contains("pacman -S --needed"),
            "install detail must use `pacman -S --needed`, got: {detail}"
        );
        assert!(
            !detail.contains("-Syu"),
            "install detail must NOT contain `-Syu` (no forced system upgrade), got: {detail}"
        );
    }
}
