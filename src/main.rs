mod cli;
mod core;
mod gui;
mod utils;

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let args = cli::Cli::parse();

    if args.needs_root() && !is_root() {
        eprintln!("This operation requires root privileges.");
        eprintln!(
            "Re-run with `sudo archgpu ...` or `pkexec archgpu ...`."
        );
        std::process::exit(1);
    }

    if args.has_any_action() {
        cli::run(args)
    } else {
        gui::run()
    }
}

fn is_root() -> bool {
    // SAFETY: geteuid() has no preconditions and returns a plain uid_t.
    unsafe { libc::geteuid() == 0 }
}
