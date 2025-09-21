use std::net::SocketAddr;

use axum::Router;
use gto_trainer::game::HeroAction;
use gto_trainer::session::{SessionState, SessionStatus};
use gto_trainer::web;
use reqwest::Client;
use serde::Serialize;
use serde_json::json;
use tokio::time::{Duration, sleep};

#[derive(Serialize)]
struct ActionPayload {
    action: HeroAction,
}

#[tokio::test]
async fn web_api_supports_session_flow() -> anyhow::Result<()> {
    let app: Router = web::router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr: SocketAddr = listener.local_addr()?;
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let base_url = format!("http://{}", addr);
    let client = Client::builder().build()?;

    sleep(Duration::from_millis(25)).await;

    let mut state: SessionState = client
        .post(format!("{}/api/sessions", base_url))
        .json(&json!({
            "hands": 1,
            "mc_samples": 80,
            "rival_style": "balanced"
        }))
        .send()
        .await?
        .json()
        .await?;

    assert_eq!(state.status, SessionStatus::AwaitingInput);
    assert!(!state.node.action_options.is_empty());

    let action = state.node.action_options[0].action.clone();

    state = client
        .post(format!(
            "{}/api/sessions/{}/actions",
            base_url, state.session_id
        ))
        .json(&ActionPayload { action })
        .send()
        .await?
        .json()
        .await?;

    assert!(matches!(
        state.status,
        SessionStatus::AwaitingInput | SessionStatus::Completed
    ));

    server.abort();
    let _ = server.await;
    Ok(())
}
