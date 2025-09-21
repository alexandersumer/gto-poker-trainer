use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use uuid::Uuid;

use crate::game::HeroAction;
use crate::rival::RivalStyle;
use crate::session::{Session, SessionConfig, SessionState};

#[derive(Clone)]
struct AppState {
    sessions: Arc<RwLock<HashMap<Uuid, Arc<Mutex<Session>>>>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn insert_session(&self, session: Session) -> Arc<Mutex<Session>> {
        let id = session.id();
        let entry = Arc::new(Mutex::new(session));
        self.sessions.write().insert(id, entry.clone());
        entry
    }

    fn get_session(&self, id: &Uuid) -> Option<Arc<Mutex<Session>>> {
        self.sessions.read().get(id).cloned()
    }
}

#[derive(Debug, Deserialize)]
struct StartSessionRequest {
    hands: Option<u32>,
    mc_samples: Option<u32>,
    seed: Option<u64>,
    rival_style: Option<RivalStyle>,
}

#[derive(Debug, Deserialize)]
struct ActionRequest {
    action: HeroAction,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, thiserror::Error)]
enum ApiError {
    #[error("session not found")]
    NotFound,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self {
            ApiError::NotFound => StatusCode::NOT_FOUND,
        };
        let body = Json(ErrorResponse {
            error: self.to_string(),
        });
        (status, body).into_response()
    }
}

pub async fn serve(addr: SocketAddr) -> Result<()> {
    let state = AppState::new();
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn build_router(state: AppState) -> Router {
    let api = Router::new()
        .route("/sessions", post(start_session))
        .route("/sessions/:id", get(fetch_session))
        .route("/sessions/:id/actions", post(apply_action));

    Router::new()
        .route("/healthz", get(health))
        .nest("/api", api)
        .nest_service("/", ServeDir::new("public"))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

pub fn router() -> Router {
    build_router(AppState::new())
}

async fn health() -> &'static str {
    "ok"
}

async fn start_session(
    State(state): State<AppState>,
    Json(req): Json<StartSessionRequest>,
) -> Result<Json<SessionState>, ApiError> {
    let config = SessionConfig {
        hands: req.hands.unwrap_or(1),
        mc_samples: req.mc_samples.unwrap_or(200),
        rival_style: req.rival_style.unwrap_or(RivalStyle::Balanced),
        seed: req.seed,
    };

    let session = Session::new(config);
    let session_arc = state.insert_session(session);
    let mut guard = session_arc.lock();
    let snapshot = guard.snapshot();
    Ok(Json(snapshot))
}

async fn fetch_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<SessionState>, ApiError> {
    let session_arc = state.get_session(&id).ok_or(ApiError::NotFound)?;
    let mut session = session_arc.lock();
    Ok(Json(session.snapshot()))
}

async fn apply_action(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<ActionRequest>,
) -> Result<Json<SessionState>, ApiError> {
    let session_arc = state.get_session(&id).ok_or(ApiError::NotFound)?;
    let mut session = session_arc.lock();
    session.apply_action(&req.action);
    Ok(Json(session.snapshot()))
}
