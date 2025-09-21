use gto_trainer::rival::RivalStyle;
use gto_trainer::trainer::{Trainer, TrainerConfig};

#[test]
fn autoplay_completes_requested_hands() {
    let config = TrainerConfig {
        hands: 2,
        mc_samples: 100,
        seed: Some(1234),
        rival_style: RivalStyle::Aggressive,
        no_color: true,
    };

    let mut trainer = Trainer::new(config);
    let summary = trainer.autoplay_best().expect("autoplay succeeds");

    assert_eq!(summary.hands_played, 2);
    assert!(summary.total_ev_loss_bb >= 0.0);
}
