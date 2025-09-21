use std::io::{self, Write};

use anyhow::Result;
use owo_colors::OwoColorize;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::game::{HeroAction, HeroActionKind};
use crate::rival::RivalStyle;
use crate::session::{Session, SessionConfig, SessionState, SessionStatus, SessionSummary};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    pub hands: u32,
    pub mc_samples: u32,
    pub seed: Option<u64>,
    #[serde(default)]
    pub rival_style: RivalStyle,
    #[serde(default)]
    pub no_color: bool,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            hands: 1,
            mc_samples: 200,
            seed: None,
            rival_style: RivalStyle::Balanced,
            no_color: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActionChoice {
    pub action: HeroAction,
    pub chosen_ev_bb: f32,
}

pub struct Trainer {
    config: TrainerConfig,
    session: Session,
    _rng: StdRng,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        let seed = config.seed.unwrap_or_else(rand::random);
        let session_config = SessionConfig {
            hands: config.hands,
            mc_samples: config.mc_samples,
            rival_style: config.rival_style,
            seed: Some(seed),
        };
        let session = Session::new(session_config);
        Self {
            config,
            session,
            _rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn run(&mut self) -> Result<()> {
        let mut input = String::new();

        loop {
            let snapshot = self.session.snapshot();
            if snapshot.status == SessionStatus::Completed {
                self.print_summary(&snapshot.summary);
                break;
            }

            self.print_node(&snapshot);

            if snapshot.node.action_options.is_empty() {
                self.session.apply_action(&HeroAction {
                    kind: HeroActionKind::Check,
                    size_bb: None,
                });
                continue;
            }

            loop {
                input.clear();
                print!(
                    "Select action [1-{}] (h=help, q=quit): ",
                    snapshot.node.action_options.len()
                );
                io::stdout().flush()?;
                io::stdin().read_line(&mut input)?;
                let trimmed = input.trim().to_lowercase();

                if trimmed == "q" {
                    self.print_summary(&snapshot.summary);
                    return Ok(());
                }

                if trimmed == "h" {
                    self.print_help(&snapshot);
                    continue;
                }

                let len = snapshot.node.action_options.len();
                match trimmed.parse::<usize>() {
                    Ok(index) if (1..=len).contains(&index) => {
                        let action = snapshot.node.action_options[index - 1].action.clone();
                        self.session.apply_action(&action);
                        break;
                    }
                    _ => {
                        println!("Invalid selection. Try again or press 'h' for help.");
                    }
                }
            }
        }

        Ok(())
    }

    pub fn session_state(&mut self) -> SessionState {
        self.session.snapshot()
    }

    pub fn apply_action(&mut self, action: HeroAction) {
        self.session.apply_action(&action);
    }

    pub fn summary(&mut self) -> SessionSummary {
        self.session.snapshot().summary
    }

    fn print_node(&self, snapshot: &SessionState) {
        let hero_cards = snapshot.node.hero_cards.join(" ");
        let board = if snapshot.node.board.is_empty() {
            "--".to_string()
        } else {
            snapshot.node.board.join(" ")
        };

        if self.config.no_color {
            println!(
                "Hand {} | Hero {} | Board {} | Pot {:.1}bb | Options: {}",
                snapshot.hand_index,
                hero_cards,
                board,
                snapshot.node.pot_bb,
                snapshot
                    .node
                    .action_options
                    .iter()
                    .enumerate()
                    .map(|(idx, opt)| format!(
                        "{}. {} ({:.2}bb)",
                        idx + 1,
                        describe_action(&opt.action),
                        opt.ev_delta_bb
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        } else {
            println!(
                "{} {} {} {} {} {} {} {:.1}bb {} {}",
                "Hand".bold().cyan(),
                snapshot.hand_index,
                "Hero".bold().white(),
                hero_cards.bold().yellow(),
                "Board".bold().white(),
                board.bold().blue(),
                "Pot".bold().white(),
                snapshot.node.pot_bb,
                "Options".bold().yellow(),
                snapshot
                    .node
                    .action_options
                    .iter()
                    .enumerate()
                    .map(|opt| format!(
                        "{}. {} ({:.2}bb)",
                        opt.0 + 1,
                        describe_action(&opt.1.action).bold().green(),
                        opt.1.ev_delta_bb
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    pub fn print_summary(&self, summary: &SessionSummary) {
        if self.config.no_color {
            println!(
                "Summary: hands={}, EV loss={:.2}bb, profit={:.2}bb",
                summary.hands_played, summary.total_ev_loss_bb, summary.total_profit_bb
            );
        } else {
            println!(
                "{} {} {} {:.2}bb {} {:.2}bb",
                "Summary".bold().magenta(),
                summary.hands_played,
                "EV loss".bold().white(),
                summary.total_ev_loss_bb,
                "Profit".bold().white(),
                summary.total_profit_bb
            );
        }
    }

    fn print_help(&self, snapshot: &SessionState) {
        println!("Available actions for hand {}:", snapshot.hand_index);
        for (idx, option) in snapshot.node.action_options.iter().enumerate() {
            println!(
                "  {}. {} ({:.2}bb) — {}",
                idx + 1,
                describe_action(&option.action),
                option.ev_delta_bb,
                option.description
            );
        }
        println!("Press the number of your choice, 'h' to view this help, or 'q' to quit.");
    }

    pub fn autoplay_best(&mut self) -> Result<SessionSummary> {
        loop {
            let snapshot = self.session.snapshot();
            if snapshot.status == SessionStatus::Completed {
                return Ok(snapshot.summary);
            }

            let best_action = snapshot
                .node
                .action_options
                .iter()
                .max_by(|a, b| a.ev_delta_bb.total_cmp(&b.ev_delta_bb))
                .map(|opt| opt.action.clone())
                .unwrap_or(HeroAction {
                    kind: HeroActionKind::Fold,
                    size_bb: None,
                });

            self.session.apply_action(&best_action);
        }
    }
}

fn describe_action(action: &HeroAction) -> String {
    match action.kind {
        HeroActionKind::Fold => "Fold".to_string(),
        HeroActionKind::Call => format!("Call {:.1}bb", action.size_bb.unwrap_or_default()),
        HeroActionKind::Check => "Check".to_string(),
        HeroActionKind::Bet => format!("Bet {:.1}bb", action.size_bb.unwrap_or_default()),
        HeroActionKind::Raise => format!("Raise to {:.1}bb", action.size_bb.unwrap_or_default()),
    }
}
