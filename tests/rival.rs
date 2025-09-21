use gto_trainer::cards::{Card, Rank, Suit};
use gto_trainer::rival::{RivalProfile, RivalStyle};

#[test]
fn rival_profile_probabilities_are_clamped() {
    let profile = RivalProfile::resolve(RivalStyle::Aggressive);
    let hero_cards = [
        Card::new(Rank::Two, Suit::Clubs),
        Card::new(Rank::Seven, Suit::Hearts),
    ];
    let strength = profile.hand_strength_hint(&hero_cards);
    assert!((0.0..=1.0).contains(&strength));

    let fold = profile.fold_to_three_bet(0.0);
    assert!((0.05..=0.85).contains(&fold));

    let bluff = profile.bluff_tendency();
    assert!(bluff > 0.0);
}

#[test]
fn rival_style_values_differ() {
    let aggressive = RivalProfile::resolve(RivalStyle::Aggressive);
    let passive = RivalProfile::resolve(RivalStyle::Passive);

    assert!(aggressive.turn_barrel_frequency > passive.turn_barrel_frequency);
    assert!(aggressive.preflop_fold_to_three_bet < passive.preflop_fold_to_three_bet);
}
