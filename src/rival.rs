use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

use crate::cards::Card;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RivalStyle {
    #[default]
    Balanced,
    Aggressive,
    Passive,
}

#[derive(Debug, Clone, Copy)]
pub struct RivalProfile {
    pub name: &'static str,
    pub preflop_fold_to_three_bet: f32,
    pub flop_continuation_bet: f32,
    pub turn_barrel_frequency: f32,
    pub river_probe_frequency: f32,
    pub aggression: f32,
}

impl RivalProfile {
    pub fn resolve(style: RivalStyle) -> Self {
        match style {
            RivalStyle::Balanced => Self {
                name: "balanced",
                preflop_fold_to_three_bet: 0.48,
                flop_continuation_bet: 0.62,
                turn_barrel_frequency: 0.52,
                river_probe_frequency: 0.33,
                aggression: 0.5,
            },
            RivalStyle::Aggressive => Self {
                name: "aggressive",
                preflop_fold_to_three_bet: 0.38,
                flop_continuation_bet: 0.71,
                turn_barrel_frequency: 0.64,
                river_probe_frequency: 0.47,
                aggression: 0.68,
            },
            RivalStyle::Passive => Self {
                name: "passive",
                preflop_fold_to_three_bet: 0.57,
                flop_continuation_bet: 0.44,
                turn_barrel_frequency: 0.36,
                river_probe_frequency: 0.21,
                aggression: 0.32,
            },
        }
    }

    pub fn fold_to_three_bet(&self, hero_strength: f32) -> f32 {
        let adjustment = (0.5 - hero_strength) * 0.35;
        (self.preflop_fold_to_three_bet + adjustment).clamp(0.05, 0.85)
    }

    pub fn continuation_bet_flop(&self) -> f32 {
        self.flop_continuation_bet
    }

    pub fn barrel_turn(&self) -> f32 {
        self.turn_barrel_frequency
    }

    pub fn probe_river(&self) -> f32 {
        self.river_probe_frequency
    }

    pub fn bluff_tendency(&self) -> f32 {
        self.aggression
    }

    pub fn random_fold<R: Rng>(&self, rng: &mut R, probability: f32) -> bool {
        let uniform = Uniform::new_inclusive(0.0f32, 1.0f32);
        uniform.sample(rng) < probability
    }

    pub fn random_decision<R: Rng>(&self, rng: &mut R, probability: f32) -> bool {
        let uniform = Uniform::new_inclusive(0.0f32, 1.0f32);
        uniform.sample(rng) < probability
    }

    pub fn describe(&self) -> &'static str {
        self.name
    }

    pub fn hand_strength_hint(&self, hero_cards: &[Card; 2]) -> f32 {
        let ranks: [u8; 2] = [hero_cards[0].rank_value(), hero_cards[1].rank_value()];
        let connectors = (ranks[0] as i8 - ranks[1] as i8).abs() <= 1;
        let pair = hero_cards[0].rank == hero_cards[1].rank;
        let suited = hero_cards[0].suit == hero_cards[1].suit;
        let base = (ranks[0] + ranks[1]) as f32 / 28.0; // normalise between 0 and ~1
        let mut strength = base;
        if pair {
            strength += 0.25;
        } else if connectors {
            strength += 0.08;
        }
        if suited {
            strength += 0.05;
        }
        strength.clamp(0.0, 1.0)
    }
}
