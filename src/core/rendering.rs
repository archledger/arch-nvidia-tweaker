// ═══════════════════════════════════════════════════════════════════════════════════════════
// Phase 22: Smart diagnostics — root-cause analysis for "why is my system in llvmpipe?"
// ═══════════════════════════════════════════════════════════════════════════════════════════
//
// On modern Arch (kernel 6.x + Mesa 25+), when GNOME / KDE falls back to software
// rendering, the cause is one of five bugs in ~99% of cases:
//
//   1. The user upgraded the kernel via pacman but hasn't rebooted. The running
//      kernel's modules directory under /usr/lib/modules/<release> was removed by the
//      upgrade, so modprobe fails on every subsequent load. Only a reboot fixes it.
//      This is uniquely an Arch-rolling-release friction point.
//
//   2. `nomodeset` is on the kernel cmdline. This is a hard lock — it blocks DRM
//      modesetting for every GPU driver, forcing llvmpipe. Usually set by a rescue
//      boot entry or by a tutorial that told the user to add it. Must be removed
//      from the cmdline source (GRUB / systemd-boot / Limine / UKI) and regenerated.
//
//   3. Secure Boot is enabled but the NVIDIA DKMS module isn't signed. The module
//      loader silently rejects unsigned kernel modules when SB is on. The fix is
//      either to disable SB in UEFI or sign modules via sbctl — neither is safely
//      automatable from the tool.
//
//   4. A Vulkan ICD JSON in /usr/share/vulkan/icd.d/ points to a `library_path` that
//      no longer exists (uninstalled package, dangling symlink, stale manual install).
//      The Vulkan loader opens the JSON, fails to dlopen, and either silently drops
//      that ICD or — worse — returns an error the caller's fallback chain converts
//      to llvmpipe.
//
//   5. The system is DEMONSTRABLY in llvmpipe right now — glxinfo or vulkaninfo
//      renderer string literally contains "llvmpipe"/"software". Best-effort probe;
//      requires mesa-utils / vulkan-tools to be installed and a compositor to be
//      running. Not reliable from a pkexec context so this one's advisory.
//
// This module provides pure, testable checks for each class. `diagnostics::scan`
// calls them and emits Findings; where the fix is safely automatable (#2 only —
// nomodeset removal), a RepairAction in core::repair picks it up.
// ═══════════════════════════════════════════════════════════════════════════════════════════

use std::path::{Path, PathBuf};

// ── Check 1: Running-kernel staleness ─────────────────────────────────────────────────────────

/// Returned when the running kernel's modules directory is missing — pacman upgraded
/// the kernel out from under us and a reboot is required to activate the new image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelStaleness {
    /// The running kernel's version string (e.g. "6.19.11-arch1-1") read from
    /// /proc/sys/kernel/osrelease.
    pub running_kernel: String,
    /// The modules directory the running kernel EXPECTS to exist but doesn't.
    pub missing_modules_dir: PathBuf,
}

/// Pure: given the path to the kernel release source and the root of the modules
/// directory tree, return staleness info if `<modules_dir>/<release>/` is missing.
/// Returns None when things are healthy (or when we couldn't read the release — no
/// signal beats a false positive here).
pub fn check_kernel_staleness(
    osrelease_path: &Path,
    modules_dir: &Path,
) -> Option<KernelStaleness> {
    let osrelease = std::fs::read_to_string(osrelease_path).ok()?;
    let running = osrelease.trim().to_string();
    if running.is_empty() {
        return None;
    }
    let expected = modules_dir.join(&running);
    if expected.exists() {
        None
    } else {
        Some(KernelStaleness {
            running_kernel: running,
            missing_modules_dir: expected,
        })
    }
}

// ── Check 2: `nomodeset` in /proc/cmdline ─────────────────────────────────────────────────────

/// True iff `nomodeset` appears as a whole token in the kernel cmdline.
/// Whole-word match ensures `nomodeset_something` or `no-nomodeset-foo` don't trigger.
pub fn check_nomodeset_in_cmdline(cmdline_path: &Path) -> bool {
    let Ok(body) = std::fs::read_to_string(cmdline_path) else {
        return false;
    };
    body.split_whitespace().any(|t| t == "nomodeset")
}

// ── Check 3: Secure Boot state via EFI variable ───────────────────────────────────────────────

/// Secure Boot firmware setting as read from /sys/firmware/efi/efivars/SecureBoot-<guid>.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecureBootStatus {
    Enabled,
    Disabled,
    /// efivars directory missing (non-UEFI boot) OR variable absent OR value unexpected.
    /// Conservative default — don't warn the user when we don't know.
    Unknown,
}

/// UEFI global-variable GUID — every firmware uses this exact UUID for the SecureBoot
/// variable. Quoted verbatim from the UEFI spec.
const SECURE_BOOT_GUID: &str = "8be4df61-93ca-11d2-aa0d-00e098032b8c";

/// Read SecureBoot state from /sys/firmware/efi/efivars/SecureBoot-<guid>.
///
/// The efivars binary format prefixes the value with a 4-byte UEFI attributes word,
/// so the actual boolean sits at byte offset 4 (0x00 = disabled, 0x01 = enabled).
/// Any other shape → Unknown (safer than guessing).
pub fn check_secure_boot(efivars_dir: &Path) -> SecureBootStatus {
    let path = efivars_dir.join(format!("SecureBoot-{SECURE_BOOT_GUID}"));
    let Ok(bytes) = std::fs::read(&path) else {
        return SecureBootStatus::Unknown;
    };
    if bytes.len() < 5 {
        return SecureBootStatus::Unknown;
    }
    match bytes[4] {
        0 => SecureBootStatus::Disabled,
        1 => SecureBootStatus::Enabled,
        _ => SecureBootStatus::Unknown,
    }
}

// ── Check 4: Vulkan ICD JSON validity ─────────────────────────────────────────────────────────

/// A detected problem with a Vulkan ICD manifest file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcdIssue {
    pub json_path: PathBuf,
    pub problem: IcdProblem,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IcdProblem {
    /// The JSON couldn't be read, or we couldn't find `library_path` inside it.
    Unparseable,
    /// `library_path` is an absolute path and the referenced file doesn't exist.
    DanglingAbsolutePath(String),
}

/// Enumerate /usr/share/vulkan/icd.d/*.json, parse each for its library_path, and
/// return any that are broken. We only validate ABSOLUTE library_path values —
/// SONAME-style values (e.g. "libGLX_nvidia.so.0") resolve through the dynamic
/// loader at dlopen time and would require actually calling dlopen to validate,
/// which would side-effect the current process. Absolute paths are the common case
/// for ICDs shipped by Arch packages.
pub fn check_vulkan_icds(icd_dir: &Path) -> Vec<IcdIssue> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(icd_dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(body) = std::fs::read_to_string(&path) else {
            out.push(IcdIssue {
                json_path: path,
                problem: IcdProblem::Unparseable,
            });
            continue;
        };
        let Some(lib_path) = extract_library_path(&body) else {
            out.push(IcdIssue {
                json_path: path,
                problem: IcdProblem::Unparseable,
            });
            continue;
        };
        if lib_path.starts_with('/') && !Path::new(&lib_path).exists() {
            out.push(IcdIssue {
                json_path: path,
                problem: IcdProblem::DanglingAbsolutePath(lib_path),
            });
        }
    }
    out
}

/// Minimal `library_path` extractor that doesn't require pulling in serde_json as a
/// direct dependency. Vulkan ICD JSONs are machine-written by driver packages and
/// well-formed; a targeted state-machine scan is enough and avoids a heavy dep for
/// a single diagnostic check.
///
/// Looks for `"library_path" : "VALUE"` with arbitrary whitespace between tokens.
pub fn extract_library_path(json: &str) -> Option<String> {
    const KEY: &str = "\"library_path\"";
    let idx = json.find(KEY)?;
    let after = &json[idx + KEY.len()..];
    let after = after.trim_start().strip_prefix(':')?.trim_start();
    let after = after.strip_prefix('"')?;
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

// ── Check 5: Direct llvmpipe probe ────────────────────────────────────────────────────────────

/// What a renderer-string probe found. Best-effort — the underlying tool (`glxinfo`
/// from mesa-utils, `vulkaninfo` from vulkan-tools) may not be installed, or may not
/// have a running compositor to query. `Unknown` is the right default in those cases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RendererState {
    /// Tool reported a hardware renderer (or at least, nothing recognizable as SW).
    Hardware(String),
    /// Tool reported `llvmpipe` / `softpipe` / `swrast` — the system is in SW rendering.
    SoftwareRendering(String),
    /// Probe tool missing OR no compositor context OR output unparseable.
    Unknown,
}

/// Parse a `glxinfo -B` or `vulkaninfo --summary` output blob for the current renderer.
/// Pure function — caller provides the text. Look for an "llvmpipe" substring
/// (case-insensitive) in any line that also says "renderer" or "deviceName" to avoid
/// false positives from unrelated output.
pub fn classify_renderer_output(output: &str) -> RendererState {
    let mut hit_renderer_line: Option<String> = None;
    for line in output.lines() {
        let lower = line.to_ascii_lowercase();
        if !(lower.contains("renderer") || lower.contains("devicename")) {
            continue;
        }
        if lower.contains("llvmpipe")
            || lower.contains("softpipe")
            || lower.contains("swrast")
            || lower.contains("software rasterizer")
        {
            return RendererState::SoftwareRendering(line.trim().to_string());
        }
        // Remember the first hardware-looking renderer line so we can return it.
        if hit_renderer_line.is_none() {
            hit_renderer_line = Some(line.trim().to_string());
        }
    }
    match hit_renderer_line {
        Some(line) => RendererState::Hardware(line),
        None => RendererState::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // ── kernel_staleness ──────────────────────────────────────────────────────────────────

    fn seed_file(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn kernel_staleness_none_when_modules_dir_present() {
        let dir = tempdir().unwrap();
        let osrel = dir.path().join("osrelease");
        let mods = dir.path().join("modules");
        seed_file(&osrel, "6.19.11-arch1-1\n");
        std::fs::create_dir_all(mods.join("6.19.11-arch1-1")).unwrap();

        assert_eq!(check_kernel_staleness(&osrel, &mods), None);
    }

    #[test]
    fn kernel_staleness_detects_missing_modules_dir() {
        let dir = tempdir().unwrap();
        let osrel = dir.path().join("osrelease");
        let mods = dir.path().join("modules");
        seed_file(&osrel, "6.19.11-arch1-1\n");
        std::fs::create_dir_all(&mods).unwrap();
        // Note: no 6.19.11-arch1-1/ subdir — simulates pacman-upgraded-since-boot.

        let stale = check_kernel_staleness(&osrel, &mods).unwrap();
        assert_eq!(stale.running_kernel, "6.19.11-arch1-1");
        assert_eq!(stale.missing_modules_dir, mods.join("6.19.11-arch1-1"));
    }

    #[test]
    fn kernel_staleness_returns_none_on_unreadable_osrelease() {
        let dir = tempdir().unwrap();
        let osrel = dir.path().join("osrelease");
        let mods = dir.path().join("modules");
        // osrel intentionally NOT created.
        assert_eq!(check_kernel_staleness(&osrel, &mods), None);
    }

    #[test]
    fn kernel_staleness_multi_kernel_linux_zen() {
        // Simulate a user running linux-zen. /usr/lib/modules has BOTH `linux` and
        // `linux-zen` kernel subdirs; osrelease reports the zen one. Staleness check
        // must key on the RUNNING release, not guess.
        let dir = tempdir().unwrap();
        let osrel = dir.path().join("osrelease");
        let mods = dir.path().join("modules");
        seed_file(&osrel, "6.19.11-zen1-1\n");
        std::fs::create_dir_all(mods.join("6.19.11-arch1-1")).unwrap();
        std::fs::create_dir_all(mods.join("6.19.11-zen1-1")).unwrap();
        assert_eq!(check_kernel_staleness(&osrel, &mods), None);

        // Now simulate pacman upgrading linux-zen out from under the running kernel:
        std::fs::remove_dir_all(mods.join("6.19.11-zen1-1")).unwrap();
        let stale = check_kernel_staleness(&osrel, &mods).unwrap();
        assert_eq!(stale.running_kernel, "6.19.11-zen1-1");
    }

    // ── nomodeset ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn nomodeset_detected_as_whole_token() {
        let dir = tempdir().unwrap();
        let cmdline = dir.path().join("cmdline");
        seed_file(&cmdline, "rw quiet nomodeset splash\n");
        assert!(check_nomodeset_in_cmdline(&cmdline));
    }

    #[test]
    fn nomodeset_not_detected_as_substring() {
        // Regression guard: `nomodesetting` or `xnomodesetx` must not count.
        let dir = tempdir().unwrap();
        let cmdline = dir.path().join("cmdline");
        seed_file(&cmdline, "rw quiet nomodesetting\n");
        assert!(!check_nomodeset_in_cmdline(&cmdline));
    }

    #[test]
    fn nomodeset_absent_on_clean_cmdline() {
        let dir = tempdir().unwrap();
        let cmdline = dir.path().join("cmdline");
        seed_file(&cmdline, "rw quiet nvidia-drm.modeset=1 nvidia-drm.fbdev=1\n");
        assert!(!check_nomodeset_in_cmdline(&cmdline));
    }

    #[test]
    fn nomodeset_missing_cmdline_file_returns_false() {
        let dir = tempdir().unwrap();
        let cmdline = dir.path().join("cmdline");
        // no seed — file doesn't exist
        assert!(!check_nomodeset_in_cmdline(&cmdline));
    }

    // ── secure_boot ───────────────────────────────────────────────────────────────────────

    #[test]
    fn secure_boot_detects_enabled() {
        let dir = tempdir().unwrap();
        let sb_path = dir
            .path()
            .join(format!("SecureBoot-{SECURE_BOOT_GUID}"));
        // 4 attribute bytes + 1 value byte (0x01 = enabled).
        std::fs::write(&sb_path, [0x07, 0x00, 0x00, 0x00, 0x01]).unwrap();
        assert_eq!(check_secure_boot(dir.path()), SecureBootStatus::Enabled);
    }

    #[test]
    fn secure_boot_detects_disabled() {
        let dir = tempdir().unwrap();
        let sb_path = dir
            .path()
            .join(format!("SecureBoot-{SECURE_BOOT_GUID}"));
        std::fs::write(&sb_path, [0x07, 0x00, 0x00, 0x00, 0x00]).unwrap();
        assert_eq!(check_secure_boot(dir.path()), SecureBootStatus::Disabled);
    }

    #[test]
    fn secure_boot_unknown_when_efivars_dir_missing() {
        let dir = tempdir().unwrap();
        // no SecureBoot-<guid> file.
        assert_eq!(check_secure_boot(dir.path()), SecureBootStatus::Unknown);
    }

    #[test]
    fn secure_boot_unknown_on_short_variable() {
        let dir = tempdir().unwrap();
        let sb_path = dir
            .path()
            .join(format!("SecureBoot-{SECURE_BOOT_GUID}"));
        std::fs::write(&sb_path, [0x07, 0x00, 0x00]).unwrap();
        assert_eq!(check_secure_boot(dir.path()), SecureBootStatus::Unknown);
    }

    // ── vulkan ICDs ───────────────────────────────────────────────────────────────────────

    const NVIDIA_ICD_JSON: &str = r#"{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "/usr/lib/libGLX_nvidia.so.0",
        "api_version" : "1.3.289"
    }
}"#;

    const RELATIVE_ICD_JSON: &str = r#"{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libvulkan_radeon.so",
        "api_version" : "1.3.278"
    }
}"#;

    #[test]
    fn extract_library_path_simple() {
        let v = extract_library_path(NVIDIA_ICD_JSON).unwrap();
        assert_eq!(v, "/usr/lib/libGLX_nvidia.so.0");
    }

    #[test]
    fn extract_library_path_handles_extra_whitespace() {
        let json = r#"{"ICD":{"library_path"     :     "/usr/lib/foo.so"}}"#;
        assert_eq!(extract_library_path(json).unwrap(), "/usr/lib/foo.so");
    }

    #[test]
    fn extract_library_path_none_when_key_missing() {
        let json = r#"{"ICD":{"api_version":"1.3"}}"#;
        assert!(extract_library_path(json).is_none());
    }

    #[test]
    fn check_vulkan_icds_empty_on_missing_dir() {
        let dir = tempdir().unwrap();
        // icd.d not created.
        assert!(check_vulkan_icds(&dir.path().join("icd.d")).is_empty());
    }

    #[test]
    fn check_vulkan_icds_skips_healthy_absolute_path() {
        let dir = tempdir().unwrap();
        let icd_dir = dir.path().join("icd.d");
        std::fs::create_dir_all(&icd_dir).unwrap();
        // Create a "real" library at the absolute path the JSON references.
        let lib_target = dir.path().join("lib/libtest.so.0");
        std::fs::create_dir_all(lib_target.parent().unwrap()).unwrap();
        std::fs::write(&lib_target, b"fake").unwrap();
        let json = format!(
            r#"{{"ICD":{{"library_path":"{}","api_version":"1.3"}}}}"#,
            lib_target.display()
        );
        std::fs::write(icd_dir.join("test_icd.json"), json).unwrap();
        assert!(check_vulkan_icds(&icd_dir).is_empty());
    }

    #[test]
    fn check_vulkan_icds_flags_dangling_absolute_path() {
        let dir = tempdir().unwrap();
        let icd_dir = dir.path().join("icd.d");
        std::fs::create_dir_all(&icd_dir).unwrap();
        let json = r#"{"ICD":{"library_path":"/nonexistent/libdangling.so.0"}}"#;
        std::fs::write(icd_dir.join("dangling.json"), json).unwrap();

        let issues = check_vulkan_icds(&icd_dir);
        assert_eq!(issues.len(), 1);
        assert!(matches!(
            &issues[0].problem,
            IcdProblem::DanglingAbsolutePath(p) if p == "/nonexistent/libdangling.so.0"
        ));
    }

    #[test]
    fn check_vulkan_icds_ignores_relative_library_path() {
        // Relative/SONAME entries resolve through the dynamic loader at dlopen time.
        // We can't validate without actually calling dlopen, which would side-effect
        // the current process. Leave them alone.
        let dir = tempdir().unwrap();
        let icd_dir = dir.path().join("icd.d");
        std::fs::create_dir_all(&icd_dir).unwrap();
        std::fs::write(icd_dir.join("radeon.json"), RELATIVE_ICD_JSON).unwrap();
        assert!(check_vulkan_icds(&icd_dir).is_empty());
    }

    #[test]
    fn check_vulkan_icds_flags_unparseable_json() {
        let dir = tempdir().unwrap();
        let icd_dir = dir.path().join("icd.d");
        std::fs::create_dir_all(&icd_dir).unwrap();
        std::fs::write(icd_dir.join("broken.json"), "this is not json at all").unwrap();

        let issues = check_vulkan_icds(&icd_dir);
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0].problem, IcdProblem::Unparseable));
    }

    // ── renderer classifier ───────────────────────────────────────────────────────────────

    #[test]
    fn classify_renderer_detects_llvmpipe_in_glxinfo() {
        let glx = "\
Extended renderer info (GLX_MESA_query_renderer):
    OpenGL renderer string: llvmpipe (LLVM 17.0.6, 256 bits)
    OpenGL version string: 4.5 (Compatibility Profile) Mesa 25.0.0
";
        assert!(matches!(
            classify_renderer_output(glx),
            RendererState::SoftwareRendering(_)
        ));
    }

    #[test]
    fn classify_renderer_detects_llvmpipe_in_vulkaninfo() {
        let vk = "\
GPU id = 0 (llvmpipe (LLVM 17.0.6, 256 bits)):
    deviceName = llvmpipe (LLVM 17.0.6, 256 bits)
    apiVersion = 1.3.289
";
        assert!(matches!(
            classify_renderer_output(vk),
            RendererState::SoftwareRendering(_)
        ));
    }

    #[test]
    fn classify_renderer_accepts_hardware_nvidia() {
        let glx = "\
OpenGL vendor string: NVIDIA Corporation
OpenGL renderer string: NVIDIA GeForce RTX 3060/PCIe/SSE2
OpenGL version string: 4.6.0 NVIDIA 595.58.03
";
        match classify_renderer_output(glx) {
            RendererState::Hardware(line) => assert!(line.contains("RTX 3060")),
            other => panic!("expected Hardware, got {other:?}"),
        }
    }

    #[test]
    fn classify_renderer_unknown_on_empty_output() {
        assert_eq!(classify_renderer_output(""), RendererState::Unknown);
    }

    #[test]
    fn classify_renderer_ignores_llvmpipe_outside_renderer_line() {
        // If "llvmpipe" appears in a NON-renderer line (e.g. a comment or extension
        // name), don't flag it as software rendering.
        let glx = "\
GL extensions: GL_EXT_llvmpipe_debug
OpenGL renderer string: NVIDIA GeForce RTX 3060
";
        assert!(matches!(
            classify_renderer_output(glx),
            RendererState::Hardware(_)
        ));
    }
}
