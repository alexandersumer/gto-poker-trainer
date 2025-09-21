use gto_trainer::game::{HeroActionKind, Street};
use gto_trainer::rival::RivalStyle;
use gto_trainer::session::{Session, SessionConfig, SessionStatus};

#[test]
fn calling_preflop_advances_to_flop() {
    let mut session = Session::new(SessionConfig {
        hands: 1,
        mc_samples: 200,
        rival_style: RivalStyle::Balanced,
        seed: Some(2025),
    });

    let pre = session.snapshot();
    assert_eq!(pre.status, SessionStatus::AwaitingInput);
    assert_eq!(pre.node.street, Street::Preflop);

    let call = pre
        .node
        .action_options
        .iter()
        .find(|opt| opt.action.kind == HeroActionKind::Call)
        .expect("call option")
        .action
        .clone();

    session.apply_action(&call);
    let flop = session.snapshot();

    assert_eq!(flop.status, SessionStatus::AwaitingInput);
    assert_eq!(flop.node.street, Street::Flop);
    assert_eq!(flop.node.board.len(), 3);
    assert_eq!(flop.summary.hands_played, 0);
}

#[test]
fn session_rolls_into_next_hand_after_completion() {
    let mut session = Session::new(SessionConfig {
        hands: 2,
        mc_samples: 150,
        rival_style: RivalStyle::Passive,
        seed: Some(11),
    });

    let first = session.snapshot();
    let fold = first
        .node
        .action_options
        .iter()
        .find(|opt| opt.action.kind == HeroActionKind::Fold)
        .expect("fold option")
        .action
        .clone();

    session.apply_action(&fold);
    let second = session.snapshot();

    assert_eq!(second.status, SessionStatus::AwaitingInput);
    assert_eq!(second.summary.hands_played, 1);
    assert_eq!(second.hand_index, 2);
}
