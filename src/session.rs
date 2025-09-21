use std::cmp::Ordering;

use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::cards::{Card, standard_deck};
use crate::equity::{best_five_card_hand, compare_strength, monte_carlo_equity};
use crate::game::{ActionOption, HeroAction, HeroActionKind, NodeSnapshot, Street};
use crate::rival::{RivalProfile, RivalStyle};

const MAX_STACK_BB: f32 = 100.0;
const OPEN_SIZES: [f32; 3] = [2.0, 2.5, 3.0];
const DEFAULT_THREE_BET: f32 = 9.0;

/// Configuration for a training session.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub hands: u32,
    pub mc_samples: u32,
    pub rival_style: RivalStyle,
    pub seed: Option<u64>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            hands: 1,
            mc_samples: 200,
            rival_style: RivalStyle::Balanced,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    AwaitingInput,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionSummary {
    pub hands_played: u32,
    pub total_ev_loss_bb: f32,
    pub total_profit_bb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: Uuid,
    pub hand_index: u32,
    pub node: NodeSnapshot,
    pub status: SessionStatus,
    pub summary: SessionSummary,
}

pub struct Session {
    id: Uuid,
    rng: StdRng,
    config: SessionConfig,
    profile: RivalProfile,
    current_hand: Option<Hand>,
    summary: SessionSummary,
}

#[derive(Debug)]
struct Hand {
    hero: [Card; 2],
    villain: [Card; 2],
    board: [Card; 5],
    open_size: f32,
    raise_size: f32,
    state: StreetState,
    options: Vec<ActionOption>,
    current_best_ev: f32,
    total_best_ev: f32,
    total_chosen_ev: f32,
    completed: bool,
}

#[derive(Debug, Clone, Copy)]
struct StreetState {
    street: Street,
    pot_bb: f32,
    hero_invested_bb: f32,
    villain_invested_bb: f32,
    board_revealed: usize,
    effective_stack_bb: f32,
}

#[derive(Debug, Clone, Copy)]
struct HandResult {
    profit_bb: f32,
    ev_loss_bb: f32,
}

#[derive(Debug)]
enum HandProgress {
    InProgress,
    Completed(HandResult),
}

impl Session {
    pub fn new(config: SessionConfig) -> Self {
        let seed = config.seed.unwrap_or_else(rand::random);
        let mut rng = StdRng::seed_from_u64(seed);
        let profile = RivalProfile::resolve(config.rival_style);
        let mut hand = Hand::new(&mut rng);
        hand.compute_options(&mut rng, profile, config.mc_samples);

        Self {
            id: Uuid::new_v4(),
            rng,
            config,
            profile,
            current_hand: Some(hand),
            summary: SessionSummary::default(),
        }
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn snapshot(&mut self) -> SessionState {
        if let Some(hand) = &mut self.current_hand {
            if hand.options.is_empty() && !hand.completed {
                hand.compute_options(&mut self.rng, self.profile, self.config.mc_samples);
            }
            SessionState {
                session_id: self.id,
                hand_index: self.summary.hands_played + 1,
                node: hand.node_snapshot(),
                status: SessionStatus::AwaitingInput,
                summary: self.summary.clone(),
            }
        } else {
            SessionState {
                session_id: self.id,
                hand_index: self.summary.hands_played,
                node: NodeSnapshot {
                    street: Street::Terminal,
                    pot_bb: 0.0,
                    effective_stack_bb: 0.0,
                    board: vec![],
                    hero_cards: vec![],
                    rival_cards_known: true,
                    action_options: vec![],
                },
                status: SessionStatus::Completed,
                summary: self.summary.clone(),
            }
        }
    }

    pub fn apply_action(&mut self, action: &HeroAction) {
        let hand = match &mut self.current_hand {
            Some(hand) => hand,
            None => return,
        };

        if hand.completed {
            return;
        }

        match hand.apply_action(action, &mut self.rng, self.profile, self.config.mc_samples) {
            HandProgress::InProgress => {
                hand.compute_options(&mut self.rng, self.profile, self.config.mc_samples);
            }
            HandProgress::Completed(result) => {
                self.summary.hands_played += 1;
                self.summary.total_ev_loss_bb += result.ev_loss_bb;
                self.summary.total_profit_bb += result.profit_bb;

                if self.summary.hands_played < self.config.hands {
                    let mut next_hand = Hand::new(&mut self.rng);
                    next_hand.compute_options(&mut self.rng, self.profile, self.config.mc_samples);
                    self.current_hand = Some(next_hand);
                } else {
                    self.current_hand = None;
                }
            }
        }
    }
}

impl Hand {
    fn new(rng: &mut StdRng) -> Self {
        let mut deck = standard_deck();
        deck.shuffle(rng);

        let hero = [deck.pop().unwrap(), deck.pop().unwrap()];
        let villain = [deck.pop().unwrap(), deck.pop().unwrap()];
        let board_cards = [
            deck.pop().unwrap(),
            deck.pop().unwrap(),
            deck.pop().unwrap(),
            deck.pop().unwrap(),
            deck.pop().unwrap(),
        ];

        let open_size = *OPEN_SIZES.choose(rng).unwrap_or(&2.5);
        let state = StreetState {
            street: Street::Preflop,
            pot_bb: open_size + 1.0,
            hero_invested_bb: 1.0,
            villain_invested_bb: open_size,
            board_revealed: 0,
            effective_stack_bb: effective_stack(1.0, open_size),
        };

        Self {
            hero,
            villain,
            board: board_cards,
            open_size,
            raise_size: DEFAULT_THREE_BET,
            state,
            options: Vec::new(),
            current_best_ev: 0.0,
            total_best_ev: 0.0,
            total_chosen_ev: 0.0,
            completed: false,
        }
    }

    fn compute_options(&mut self, rng: &mut StdRng, profile: RivalProfile, samples: u32) {
        let options = match self.state.street {
            Street::Preflop => self.compute_preflop_options(rng, profile, samples),
            Street::Flop | Street::Turn | Street::River => {
                self.compute_postflop_options(rng, profile, samples)
            }
            Street::Showdown | Street::Terminal => Vec::new(),
        };

        let best = options
            .iter()
            .map(|opt| opt.ev_delta_bb)
            .fold(f32::NEG_INFINITY, f32::max);
        self.current_best_ev = if best.is_finite() { best } else { 0.0 };
        self.options = options;
    }

    fn compute_preflop_options(
        &self,
        rng: &mut StdRng,
        profile: RivalProfile,
        samples: u32,
    ) -> Vec<ActionOption> {
        let hero_cards: [Card; 2] = self.hero;
        let hero_strength = profile.hand_strength_hint(&hero_cards);
        let equity = monte_carlo_equity(&self.hero, None, &[], samples, rng);
        let call_cost = (self.open_size - self.state.hero_invested_bb).max(0.0);
        let pot_after_call = 2.0 * self.open_size;
        let call_ev = equity * pot_after_call - (1.0 - equity) * call_cost;

        let pot_before_raise = self.state.pot_bb;
        let raise_to = self.raise_size;
        let raise_cost = (raise_to - self.state.hero_invested_bb).max(0.0);
        let pot_when_called = 2.0 * raise_to;
        let fold_prob = profile.fold_to_three_bet(hero_strength);
        let raise_ev = fold_prob * pot_before_raise
            + (1.0 - fold_prob) * (equity * pot_when_called - (1.0 - equity) * raise_cost);

        vec![
            ActionOption {
                action: HeroAction {
                    kind: HeroActionKind::Fold,
                    size_bb: None,
                },
                ev_delta_bb: -self.state.hero_invested_bb,
                description: "Fold and surrender the blind".to_string(),
            },
            ActionOption {
                action: HeroAction {
                    kind: HeroActionKind::Call,
                    size_bb: Some(self.open_size),
                },
                ev_delta_bb: call_ev,
                description: format!(
                    "Flat call {:.1}bb open (equity {:.1}%)",
                    self.open_size,
                    equity * 100.0
                ),
            },
            ActionOption {
                action: HeroAction {
                    kind: HeroActionKind::Raise,
                    size_bb: Some(raise_to),
                },
                ev_delta_bb: raise_ev,
                description: format!(
                    "3-bet to {:.1}bb (fold equity {:.0}%)",
                    raise_to,
                    fold_prob * 100.0
                ),
            },
        ]
    }

    fn compute_postflop_options(
        &self,
        rng: &mut StdRng,
        profile: RivalProfile,
        samples: u32,
    ) -> Vec<ActionOption> {
        let board = self.visible_board();
        let equity = monte_carlo_equity(&self.hero, None, &board, samples, rng);
        let pot = self.state.pot_bb;
        let check_ev = (2.0 * equity - 1.0) * pot;

        let bet_multiplier = match self.state.street {
            Street::Flop => 0.5,
            Street::Turn => 0.6,
            Street::River => 0.75,
            _ => 0.5,
        };
        let mut bet_size = (pot * bet_multiplier).max(0.5);
        bet_size = bet_size.min(self.state.effective_stack_bb.max(0.0));
        if bet_size < 0.5 {
            bet_size = self.state.effective_stack_bb.max(0.0);
        }

        let fold_prob = fold_probability(profile, equity, self.state.street);
        let bet_ev = fold_prob * pot
            + (1.0 - fold_prob) * (equity * (pot + 2.0 * bet_size) - (1.0 - equity) * bet_size);

        vec![
            ActionOption {
                action: HeroAction {
                    kind: HeroActionKind::Check,
                    size_bb: None,
                },
                ev_delta_bb: check_ev,
                description: format!("Check and realise equity ({:.1}% share)", equity * 100.0),
            },
            ActionOption {
                action: HeroAction {
                    kind: HeroActionKind::Bet,
                    size_bb: Some(bet_size),
                },
                ev_delta_bb: bet_ev,
                description: format!(
                    "Bet {:.1}bb ({:.0}% fold equity)",
                    bet_size,
                    fold_prob * 100.0
                ),
            },
        ]
    }

    fn apply_action(
        &mut self,
        action: &HeroAction,
        rng: &mut StdRng,
        profile: RivalProfile,
        samples: u32,
    ) -> HandProgress {
        let chosen = match self
            .options
            .iter()
            .find(|candidate| candidate.action == *action)
        {
            Some(opt) => opt.clone(),
            None => return HandProgress::InProgress,
        };

        self.total_best_ev += self.current_best_ev;
        self.total_chosen_ev += chosen.ev_delta_bb;

        match self.state.street {
            Street::Preflop => self.apply_preflop(action, chosen, rng, profile),
            Street::Flop | Street::Turn => {
                self.apply_postflop(action, chosen, rng, profile, samples)
            }
            Street::River => self.apply_river(action, chosen, rng, profile, samples),
            Street::Showdown | Street::Terminal => HandProgress::Completed(HandResult {
                profit_bb: 0.0,
                ev_loss_bb: self.current_ev_loss(),
            }),
        }
    }

    fn apply_preflop(
        &mut self,
        action: &HeroAction,
        option: ActionOption,
        rng: &mut StdRng,
        profile: RivalProfile,
    ) -> HandProgress {
        match action.kind {
            HeroActionKind::Fold => self.finish(-self.state.hero_invested_bb),
            HeroActionKind::Call => {
                let call_cost = (self.open_size - self.state.hero_invested_bb).max(0.0);
                self.state.hero_invested_bb += call_cost;
                self.refresh_state();
                self.advance_street(Street::Flop);
                HandProgress::InProgress
            }
            HeroActionKind::Raise => {
                let raise_to = option.action.size_bb.unwrap_or(self.raise_size);
                let raise_cost = (raise_to - self.state.hero_invested_bb).max(0.0);
                self.state.hero_invested_bb += raise_cost;
                self.refresh_state();

                let hero_cards: [Card; 2] = self.hero;
                let hero_strength = profile.hand_strength_hint(&hero_cards);
                let fold_prob = profile.fold_to_three_bet(hero_strength);
                if profile.random_fold(rng, fold_prob) {
                    self.finish(self.state.villain_invested_bb)
                } else {
                    let call_cost = (raise_to - self.open_size).max(0.0);
                    self.state.villain_invested_bb += call_cost;
                    self.refresh_state();
                    self.advance_street(Street::Flop);
                    HandProgress::InProgress
                }
            }
            _ => HandProgress::InProgress,
        }
    }

    fn apply_postflop(
        &mut self,
        action: &HeroAction,
        option: ActionOption,
        rng: &mut StdRng,
        profile: RivalProfile,
        samples: u32,
    ) -> HandProgress {
        match action.kind {
            HeroActionKind::Check => {
                let next = match self.state.street {
                    Street::Flop => Street::Turn,
                    Street::Turn => Street::River,
                    _ => Street::River,
                };
                self.advance_street(next);
                HandProgress::InProgress
            }
            HeroActionKind::Bet => {
                let mut bet_size = option.action.size_bb.unwrap_or(0.0);
                if bet_size <= 0.0 {
                    bet_size = (self.state.pot_bb * 0.5).max(0.5);
                }
                bet_size = bet_size.min(self.state.effective_stack_bb.max(0.0));
                self.state.hero_invested_bb += bet_size;
                self.refresh_state();

                let board = self.visible_board();
                let equity = monte_carlo_equity(&self.hero, None, &board, samples, rng);
                let fold_prob = fold_probability(profile, equity, self.state.street);

                if profile.random_fold(rng, fold_prob) {
                    self.finish(self.state.villain_invested_bb)
                } else {
                    let call_size =
                        bet_size.min((MAX_STACK_BB - self.state.villain_invested_bb).max(0.0));
                    self.state.villain_invested_bb += call_size;
                    self.refresh_state();
                    let next = match self.state.street {
                        Street::Flop => Street::Turn,
                        Street::Turn => Street::River,
                        _ => Street::River,
                    };
                    self.advance_street(next);
                    HandProgress::InProgress
                }
            }
            _ => HandProgress::InProgress,
        }
    }

    fn apply_river(
        &mut self,
        action: &HeroAction,
        option: ActionOption,
        rng: &mut StdRng,
        profile: RivalProfile,
        samples: u32,
    ) -> HandProgress {
        match action.kind {
            HeroActionKind::Check => self.resolve_showdown(),
            HeroActionKind::Bet => {
                let mut bet_size = option.action.size_bb.unwrap_or(0.0);
                if bet_size <= 0.0 {
                    bet_size = (self.state.pot_bb * 0.75).max(0.5);
                }
                bet_size = bet_size.min(self.state.effective_stack_bb.max(0.0));
                self.state.hero_invested_bb += bet_size;
                self.refresh_state();

                let board = self.visible_board();
                let equity = monte_carlo_equity(&self.hero, None, &board, samples, rng);
                let fold_prob = fold_probability(profile, equity, Street::River);

                if profile.random_fold(rng, fold_prob) {
                    self.finish(self.state.villain_invested_bb)
                } else {
                    let call_size =
                        bet_size.min((MAX_STACK_BB - self.state.villain_invested_bb).max(0.0));
                    self.state.villain_invested_bb += call_size;
                    self.refresh_state();
                    self.resolve_showdown()
                }
            }
            _ => HandProgress::InProgress,
        }
    }

    fn resolve_showdown(&mut self) -> HandProgress {
        self.state.board_revealed = 5;
        self.state.street = Street::Showdown;
        let mut board_cards = Vec::with_capacity(5);
        board_cards.extend_from_slice(&self.board);

        let mut hero_cards = Vec::with_capacity(7);
        hero_cards.extend_from_slice(&self.hero);
        hero_cards.extend_from_slice(&board_cards);

        let mut villain_cards = Vec::with_capacity(7);
        villain_cards.extend_from_slice(&self.villain);
        villain_cards.extend_from_slice(&board_cards);

        let hero_strength = best_five_card_hand(&hero_cards);
        let villain_strength = best_five_card_hand(&villain_cards);

        let profit = match compare_strength(hero_strength, villain_strength) {
            Ordering::Greater => self.state.villain_invested_bb,
            Ordering::Less => -self.state.hero_invested_bb,
            Ordering::Equal => (self.state.villain_invested_bb - self.state.hero_invested_bb) / 2.0,
        };

        self.finish(profit)
    }

    fn node_snapshot(&self) -> NodeSnapshot {
        NodeSnapshot {
            street: self.state.street,
            pot_bb: self.state.pot_bb,
            effective_stack_bb: self.state.effective_stack_bb.max(0.0),
            board: self
                .board
                .iter()
                .take(self.state.board_revealed)
                .map(|c| c.to_string())
                .collect(),
            hero_cards: self.hero.iter().map(|c| c.to_string()).collect(),
            rival_cards_known: self.completed,
            action_options: self.options.clone(),
        }
    }

    fn visible_board(&self) -> Vec<Card> {
        self.board
            .iter()
            .take(self.state.board_revealed)
            .copied()
            .collect()
    }

    fn advance_street(&mut self, target: Street) {
        self.state.street = target;
        self.state.board_revealed = match target {
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
            Street::Showdown | Street::Terminal => 5,
            Street::Preflop => 0,
        };
        self.refresh_state();
    }

    fn refresh_state(&mut self) {
        self.state.pot_bb = self.state.hero_invested_bb + self.state.villain_invested_bb;
        self.state.effective_stack_bb =
            effective_stack(self.state.hero_invested_bb, self.state.villain_invested_bb);
    }

    fn current_ev_loss(&self) -> f32 {
        (self.total_best_ev - self.total_chosen_ev).max(0.0)
    }

    fn finish(&mut self, profit_bb: f32) -> HandProgress {
        self.completed = true;
        self.state.street = Street::Terminal;
        HandProgress::Completed(HandResult {
            profit_bb,
            ev_loss_bb: self.current_ev_loss(),
        })
    }
}

fn effective_stack(hero_invested: f32, villain_invested: f32) -> f32 {
    let hero_remaining = (MAX_STACK_BB - hero_invested).max(0.0);
    let villain_remaining = (MAX_STACK_BB - villain_invested).max(0.0);
    hero_remaining.min(villain_remaining)
}

fn fold_probability(profile: RivalProfile, equity: f32, street: Street) -> f32 {
    let (base, aggression_metric) = match street {
        Street::Flop => (0.4, profile.continuation_bet_flop()),
        Street::Turn => (0.35, profile.barrel_turn()),
        Street::River => (0.3, profile.probe_river()),
        _ => (0.45, 0.5),
    };

    let aggression_adjust = (0.5 - aggression_metric) * 0.3;
    let equity_adjust = (0.5 - equity) * 0.35;
    let raw = base + aggression_adjust + equity_adjust;
    raw.clamp(0.05, 0.9)
}
