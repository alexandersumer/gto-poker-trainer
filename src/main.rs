use std::net::SocketAddr;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use gto_trainer::rival::RivalStyle;
use gto_trainer::web;
use gto_trainer::{Trainer, TrainerConfig};

#[derive(Debug, Parser)]
#[command(
    name = "gto-trainer",
    version,
    about = "Heads-up NLHE trainer (Rust edition)",
    author
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Number of hands to play (defaults to 1)
    #[arg(long, default_value_t = 1)]
    hands: u32,

    /// RNG seed (random if omitted)
    #[arg(long)]
    seed: Option<u64>,

    /// Monte Carlo samples per decision
    #[arg(long = "mc", default_value_t = 200)]
    mc_samples: u32,

    /// Disable ANSI colors in CLI output
    #[arg(long = "no-color", default_value_t = false)]
    no_color: bool,

    /// Rival style preset
    #[arg(long = "rival-style", default_value = "balanced")]
    rival_style: RivalStyleArg,

    /// Auto-play hands using the best-EV action (useful for smoke tests)
    #[arg(long, default_value_t = false)]
    auto: bool,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run the web server
    Serve {
        /// Address to bind (HOST:PORT)
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: String,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum RivalStyleArg {
    Balanced,
    Aggressive,
    Passive,
}

impl From<RivalStyleArg> for RivalStyle {
    fn from(arg: RivalStyleArg) -> Self {
        match arg {
            RivalStyleArg::Balanced => RivalStyle::Balanced,
            RivalStyleArg::Aggressive => RivalStyle::Aggressive,
            RivalStyleArg::Passive => RivalStyle::Passive,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = color_eyre::install();
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Serve { addr }) => run_server(addr).await?,
        None => run_cli(cli).await?,
    }

    Ok(())
}

async fn run_cli(cli: Cli) -> Result<()> {
    let config = TrainerConfig {
        hands: cli.hands,
        mc_samples: cli.mc_samples,
        seed: cli.seed,
        rival_style: cli.rival_style.clone().into(),
        no_color: cli.no_color,
    };
    let mut trainer = Trainer::new(config);
    if cli.auto {
        let summary = trainer.autoplay_best()?;
        trainer.print_summary(&summary);
        Ok(())
    } else {
        trainer.run()
    }
}

async fn run_server(addr: String) -> Result<()> {
    let addr: SocketAddr = addr.parse()?;
    web::serve(addr).await
}
