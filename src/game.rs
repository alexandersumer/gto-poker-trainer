use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
    Terminal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HeroActionKind {
    Fold,
    Call,
    Check,
    Bet,
    Raise,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeroAction {
    pub kind: HeroActionKind,
    pub size_bb: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionOption {
    pub action: HeroAction,
    pub ev_delta_bb: f32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeSnapshot {
    pub street: Street,
    pub pot_bb: f32,
    pub effective_stack_bb: f32,
    pub board: Vec<String>,
    pub hero_cards: Vec<String>,
    pub rival_cards_known: bool,
    pub action_options: Vec<ActionOption>,
}
