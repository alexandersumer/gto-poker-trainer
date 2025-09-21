use gto_trainer::game::HeroActionKind;
use gto_trainer::rival::RivalStyle;
use gto_trainer::session::{Session, SessionConfig, SessionStatus};

#[test]
fn folding_ends_session_and_records_summary() {
    let config = SessionConfig {
        hands: 1,
        mc_samples: 100,
        rival_style: RivalStyle::Balanced,
        seed: Some(42),
    };

    let mut session = Session::new(config);
    let initial = session.snapshot();
    assert_eq!(initial.status, SessionStatus::AwaitingInput);
    assert_eq!(initial.hand_index, 1);
    assert!(!initial.node.action_options.is_empty());

    let fold_action = initial
        .node
        .action_options
        .iter()
        .find(|opt| opt.action.kind == HeroActionKind::Fold)
        .expect("fold option available")
        .action
        .clone();

    session.apply_action(&fold_action);
    let after = session.snapshot();
    assert_eq!(after.status, SessionStatus::Completed);
    assert_eq!(after.summary.hands_played, 1);
    assert!(after.summary.total_ev_loss_bb >= 0.0);
}
