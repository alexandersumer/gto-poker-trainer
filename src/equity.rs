use std::cmp::Ordering;

use itertools::Itertools;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::cards::{Card, Rank};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum HandCategory {
    HighCard = 0,
    OnePair = 1,
    TwoPair = 2,
    ThreeOfAKind = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfAKind = 7,
    StraightFlush = 8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandStrength {
    pub category: HandCategory,
    pub ranks: [u8; 5],
}

impl PartialOrd for HandStrength {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HandStrength {
    fn cmp(&self, other: &Self) -> Ordering {
        self.category
            .cmp(&other.category)
            .then_with(|| self.ranks.cmp(&other.ranks))
    }
}

fn fill(mut values: Vec<u8>) -> [u8; 5] {
    values.resize(5, 0);
    [values[0], values[1], values[2], values[3], values[4]]
}

fn evaluate_five(cards: &[Card; 5]) -> HandStrength {
    let mut counts = [0u8; 15];
    let mut suits = [0u8; 4];
    let mut sorted_cards: Vec<u8> = cards.iter().map(|c| c.rank_value()).collect();
    sorted_cards.sort_unstable_by(|a, b| b.cmp(a));

    for card in cards {
        counts[card.rank_value() as usize] += 1;
        suits[match card.suit {
            crate::cards::Suit::Clubs => 0,
            crate::cards::Suit::Diamonds => 1,
            crate::cards::Suit::Hearts => 2,
            crate::cards::Suit::Spades => 3,
        }] += 1;
    }

    let is_flush = suits.contains(&5);

    let mut mask: u32 = 0;
    for rank_value in 2u8..=14 {
        if counts[rank_value as usize] > 0 {
            mask |= 1 << rank_value as u32;
            if rank_value == Rank::Ace.value() {
                mask |= 1 << 1; // Ace-low straight support
            }
        }
    }

    let mut straight_high = None;
    for high in (5u8..=14).rev() {
        let mut needed = 0u32;
        for i in 0..5u8 {
            let bit = (high - i) as u32;
            needed |= 1 << bit;
        }
        if mask & needed == needed {
            straight_high = Some(high);
            break;
        }
    }

    let mut groups: Vec<(u8, u8)> = (2..=14)
        .filter_map(|rank| {
            let count = counts[rank as usize];
            if count > 0 {
                Some((count, rank as u8))
            } else {
                None
            }
        })
        .collect();
    groups.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

    if is_flush && let Some(high) = straight_high {
        return HandStrength {
            category: HandCategory::StraightFlush,
            ranks: fill(vec![high, high - 1, high - 2, high - 3, high - 4]),
        };
    }

    if let Some(&(count, rank)) = groups.first() {
        match count {
            4 => {
                let kicker = groups
                    .iter()
                    .find(|(c, _)| *c == 1)
                    .map(|(_, r)| *r)
                    .unwrap_or(0);
                return HandStrength {
                    category: HandCategory::FourOfAKind,
                    ranks: fill(vec![rank, kicker]),
                };
            }
            3 => {
                if groups.get(1).map(|(c, _)| *c == 2).unwrap_or(false) {
                    let pair_rank = groups[1].1;
                    return HandStrength {
                        category: HandCategory::FullHouse,
                        ranks: fill(vec![rank, pair_rank]),
                    };
                }
            }
            _ => {}
        }
    }

    if is_flush {
        return HandStrength {
            category: HandCategory::Flush,
            ranks: fill(sorted_cards.clone()),
        };
    }

    if let Some(high) = straight_high {
        return HandStrength {
            category: HandCategory::Straight,
            ranks: fill(vec![high, high - 1, high - 2, high - 3, high - 4]),
        };
    }

    if let Some(&(count, rank)) = groups.first() {
        match count {
            3 => {
                let mut kickers: Vec<u8> = groups
                    .iter()
                    .filter(|(c, _)| *c == 1)
                    .map(|(_, r)| *r)
                    .collect();
                kickers.sort_unstable_by(|a, b| b.cmp(a));
                let mut values = vec![rank];
                values.extend(kickers);
                return HandStrength {
                    category: HandCategory::ThreeOfAKind,
                    ranks: fill(values),
                };
            }
            2 => {
                if groups.get(1).map(|(c, _)| *c == 2).unwrap_or(false) {
                    let first_pair = rank;
                    let second_pair = groups[1].1;
                    let kicker = groups
                        .iter()
                        .find(|(c, _)| *c == 1)
                        .map(|(_, r)| *r)
                        .unwrap_or(0);
                    return HandStrength {
                        category: HandCategory::TwoPair,
                        ranks: fill(vec![first_pair, second_pair, kicker]),
                    };
                } else {
                    let mut kickers: Vec<u8> = groups
                        .iter()
                        .filter(|(c, _)| *c == 1)
                        .map(|(_, r)| *r)
                        .collect();
                    kickers.sort_unstable_by(|a, b| b.cmp(a));
                    let mut values = vec![rank];
                    values.extend(kickers);
                    return HandStrength {
                        category: HandCategory::OnePair,
                        ranks: fill(values),
                    };
                }
            }
            _ => {}
        }
    }

    HandStrength {
        category: HandCategory::HighCard,
        ranks: fill(sorted_cards),
    }
}

pub fn best_five_card_hand(cards: &[Card]) -> HandStrength {
    assert!(cards.len() >= 5, "at least 5 cards required");
    cards
        .iter()
        .copied()
        .combinations(5)
        .map(|combo| {
            let arr = [combo[0], combo[1], combo[2], combo[3], combo[4]];
            evaluate_five(&arr)
        })
        .max()
        .expect("combinations non-empty")
}

pub fn compare_strength(a: HandStrength, b: HandStrength) -> Ordering {
    a.cmp(&b)
}

pub fn monte_carlo_equity<R: Rng + ?Sized>(
    hero: &[Card],
    villain: Option<&[Card]>,
    board_known: &[Card],
    samples: u32,
    rng: &mut R,
) -> f32 {
    assert!(hero.len() == 2, "hero must have two cards");
    let samples = samples.max(1);
    let mut equity_sum = 0.0f32;

    for _ in 0..samples {
        let mut deck = crate::cards::standard_deck();
        for card in hero {
            deck.retain(|c| c != card);
        }
        if let Some(villain_cards) = villain {
            for card in villain_cards {
                deck.retain(|c| c != card);
            }
        }
        for card in board_known {
            deck.retain(|c| c != card);
        }

        deck.shuffle(rng);

        let villain_cards: [Card; 2] = if let Some(villain_cards) = villain {
            [villain_cards[0], villain_cards[1]]
        } else {
            let first = deck.pop().expect("deck not empty");
            let second = deck.pop().expect("deck not empty");
            [first, second]
        };

        let mut board = board_known.to_vec();
        let cards_needed = 5usize.saturating_sub(board.len());
        for _ in 0..cards_needed {
            board.push(deck.pop().expect("cards remain"));
        }

        let hero_cards: Vec<Card> = hero.iter().copied().chain(board.iter().copied()).collect();
        let villain_cards_full: Vec<Card> = villain_cards
            .iter()
            .copied()
            .chain(board.iter().copied())
            .collect();

        let hero_strength = best_five_card_hand(&hero_cards);
        let villain_strength = best_five_card_hand(&villain_cards_full);

        match compare_strength(hero_strength, villain_strength) {
            Ordering::Greater => equity_sum += 1.0,
            Ordering::Equal => equity_sum += 0.5,
            Ordering::Less => {}
        }
    }

    equity_sum / samples as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{Card, Rank, Suit};

    #[test]
    fn straight_flush_beats_four_kind() {
        let sf = [
            Card::new(Rank::Ten, Suit::Hearts),
            Card::new(Rank::Jack, Suit::Hearts),
            Card::new(Rank::Queen, Suit::Hearts),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::Ace, Suit::Hearts),
        ];
        let four = [
            Card::new(Rank::Nine, Suit::Clubs),
            Card::new(Rank::Nine, Suit::Diamonds),
            Card::new(Rank::Nine, Suit::Hearts),
            Card::new(Rank::Nine, Suit::Spades),
            Card::new(Rank::Ace, Suit::Clubs),
        ];

        assert!(evaluate_five(&sf) > evaluate_five(&four));
    }

    #[test]
    fn wheel_straight_detected() {
        let hand = [
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Two, Suit::Diamonds),
            Card::new(Rank::Three, Suit::Hearts),
            Card::new(Rank::Four, Suit::Spades),
            Card::new(Rank::Five, Suit::Clubs),
        ];
        let strength = evaluate_five(&hand);
        assert_eq!(strength.category, HandCategory::Straight);
        assert_eq!(strength.ranks[0], 5);
    }
}
