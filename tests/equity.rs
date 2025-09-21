use rand::SeedableRng;
use rand::rngs::StdRng;

use gto_trainer::cards::{Card, Rank, Suit};
use gto_trainer::equity::{HandCategory, best_five_card_hand, monte_carlo_equity};

#[test]
fn quads_outrank_full_house() {
    let cards = vec![
        Card::new(Rank::Nine, Suit::Clubs),
        Card::new(Rank::Nine, Suit::Diamonds),
        Card::new(Rank::Nine, Suit::Hearts),
        Card::new(Rank::Nine, Suit::Spades),
        Card::new(Rank::Ace, Suit::Clubs),
        Card::new(Rank::Ace, Suit::Hearts),
        Card::new(Rank::Five, Suit::Clubs),
    ];

    let strength = best_five_card_hand(&cards);
    assert_eq!(strength.category, HandCategory::FourOfAKind);
    assert_eq!(strength.ranks[0], Rank::Nine as u8);
}

#[test]
fn monte_carlo_equity_matches_expected_range() {
    let hero = [
        Card::new(Rank::Ace, Suit::Spades),
        Card::new(Rank::Ace, Suit::Hearts),
    ];
    let villain = [
        Card::new(Rank::King, Suit::Spades),
        Card::new(Rank::King, Suit::Hearts),
    ];

    let board: Vec<Card> = Vec::new();
    let mut rng = StdRng::seed_from_u64(99);
    let equity = monte_carlo_equity(&hero, Some(&villain), &board, 10_000, &mut rng);

    // AA vs KK preflop equity is ~82%. Allow a small tolerance due to Monte Carlo variance.
    assert!(equity > 0.79 && equity < 0.85, "equity={equity}");
}

#[test]
fn monte_carlo_respects_known_board() {
    let hero = [
        Card::new(Rank::Ace, Suit::Spades),
        Card::new(Rank::King, Suit::Spades),
    ];
    let villain = [
        Card::new(Rank::Queen, Suit::Clubs),
        Card::new(Rank::Jack, Suit::Clubs),
    ];
    // Hero flopped the nuts on T-Q-J with royal draw; equity should be near 100%.
    let board = vec![
        Card::new(Rank::Ten, Suit::Spades),
        Card::new(Rank::Queen, Suit::Spades),
        Card::new(Rank::Jack, Suit::Spades),
    ];
    let mut rng = StdRng::seed_from_u64(7);
    let equity = monte_carlo_equity(&hero, Some(&villain), &board, 5_000, &mut rng);
    assert!(equity > 0.97, "equity={equity}");
}
