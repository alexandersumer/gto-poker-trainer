use std::fmt::{Display, Formatter};
use std::str::FromStr;

use rand::Rng;
use rand::seq::SliceRandom;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

impl Suit {
    pub const ALL: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

    pub fn symbol(self) -> &'static str {
        match self {
            Suit::Clubs => "♣",
            Suit::Diamonds => "♦",
            Suit::Hearts => "♥",
            Suit::Spades => "♠",
        }
    }
}

impl Display for Suit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.symbol())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(u8)]
pub enum Rank {
    Two = 2,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

impl Rank {
    pub const ALL: [Rank; 13] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];

    pub fn value(self) -> u8 {
        self as u8
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Rank::Two => "2",
            Rank::Three => "3",
            Rank::Four => "4",
            Rank::Five => "5",
            Rank::Six => "6",
            Rank::Seven => "7",
            Rank::Eight => "8",
            Rank::Nine => "9",
            Rank::Ten => "T",
            Rank::Jack => "J",
            Rank::Queen => "Q",
            Rank::King => "K",
            Rank::Ace => "A",
        }
    }
}

impl Display for Rank {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.short_label())
    }
}

impl FromStr for Rank {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "2" => Ok(Rank::Two),
            "3" => Ok(Rank::Three),
            "4" => Ok(Rank::Four),
            "5" => Ok(Rank::Five),
            "6" => Ok(Rank::Six),
            "7" => Ok(Rank::Seven),
            "8" => Ok(Rank::Eight),
            "9" => Ok(Rank::Nine),
            "T" | "10" => Ok(Rank::Ten),
            "J" => Ok(Rank::Jack),
            "Q" => Ok(Rank::Queen),
            "K" => Ok(Rank::King),
            "A" => Ok(Rank::Ace),
            _ => Err(format!("Invalid rank '{s}'")),
        }
    }
}

impl FromStr for Suit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "c" | "♣" => Ok(Suit::Clubs),
            "d" | "♦" => Ok(Suit::Diamonds),
            "h" | "♥" => Ok(Suit::Hearts),
            "s" | "♠" => Ok(Suit::Spades),
            _ => Err(format!("Invalid suit '{s}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl Card {
    pub fn new(rank: Rank, suit: Suit) -> Self {
        Self { rank, suit }
    }

    pub fn rank_value(&self) -> u8 {
        self.rank.value()
    }

    pub fn notation(&self) -> String {
        format!("{}{}", self.rank.short_label(), self.suit.symbol())
    }
}

impl Display for Card {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.notation())
    }
}

pub fn standard_deck() -> Vec<Card> {
    let mut cards = Vec::with_capacity(52);
    for suit in Suit::ALL {
        for rank in Rank::ALL {
            cards.push(Card::new(rank, suit));
        }
    }
    cards
}

pub fn deal_unique_cards<R: Rng>(rng: &mut R, count: usize) -> Vec<Card> {
    let mut deck = standard_deck();
    deck.shuffle(rng);
    deck.into_iter().take(count).collect()
}

pub fn shuffle_deck<R: Rng>(rng: &mut R, deck: &mut [Card]) {
    deck.shuffle(rng);
}
