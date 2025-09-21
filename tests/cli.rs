use assert_cmd::prelude::*;
use std::process::Command;

#[test]
fn cli_auto_mode_runs_to_completion() {
    let mut cmd = Command::cargo_bin("gto-trainer").expect("binary exists");
    cmd.arg("--hands")
        .arg("1")
        .arg("--mc")
        .arg("50")
        .arg("--no-color")
        .arg("--auto");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Summary"));
}
