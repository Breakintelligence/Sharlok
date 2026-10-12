"""Sharlok FastAPI Application - AI Security Architecture Review Interface.

Provides:
- File upload endpoint for architecture diagrams
- Full security architecture review (STRIDE is one section, not the whole scope)
- Security recommendations
- Interactive chat about the review
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from threat_modeling import (
    analyze_architecture,
    save_threat_model,
    generate_recommendations,
    threat_model_chat,
    ChatMessage,
)
from ready import ensure_ollama_ready

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("Sharlok.api")

# Background task executor for knowledge base
_kb_executor = ThreadPoolExecutor(max_workers=1)

# Directories
UPLOAD_DIR = "uploads"
THREAT_MODELS_DIR = "threat_models"
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(THREAT_MODELS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Sharlok", description="AI Security Architecture Reviewer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ===== Request/Response Models =====

class ThreatModelRequest(BaseModel):
    """Request to generate threat model from uploaded image."""
    session_id: str
    filename: str
    use_web_search: bool = False


class RecommendationRequest(BaseModel):
    """Request to generate security recommendations."""
    session_id: str
    threat_model: str


class ChatRequest(BaseModel):
    """Request to chat about threat model."""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Chat response."""
    role: str
    content: str


class ThreatModelResponse(BaseModel):
    """Threat model generation response."""
    session_id: str
    threat_model: str
    saved_path: str
    image_url: str


# ===== Session Management =====

sessions: dict = {}  # session_id → {threat_model, recommendations, chat_history, image_path}


def create_session() -> str:
    """Create a new threat modeling session."""
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        "threat_model": "",
        "recommendations": "",
        "chat_history": [],
        "image_path": "",
        "saved_path": "",
    }
    log.info("Created session: %s", session_id)
    return session_id


def get_session(session_id: str) -> dict:
    """Get session data or raise error."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


def cleanup_session(session_id: str):
    """Clean up session and temporary files."""
    if session_id in sessions:
        session = sessions[session_id]
        # Delete uploaded image
        if session.get("image_path") and os.path.exists(session["image_path"]):
            os.remove(session["image_path"])
        del sessions[session_id]
        log.info("Cleaned up session: %s", session_id)


# ===== API Endpoints =====

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI page."""
    template_path = Path(TEMPLATES_DIR) / "index.html"
    if not template_path.exists():
        return "<h1>Sharlok AI Security Architecture Reviewer</h1><p>Upload an architecture diagram to get started.</p>"
    with open(template_path, "r") as f:
        return f.read()


@app.post("/api/session")
async def new_session() -> dict:
    """Create a new threat modeling session."""
    session_id = create_session()
    return {"session_id": session_id}


@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
) -> dict:
    """Upload an architecture artifact: image, PDF, PPTX, or Mermaid/text file.

    Returns:
        Session ID and (for images) a preview URL
    """
    from file_ingest import SUPPORTED_EXTS, is_image

    if not session_id:
        session_id = create_session()

    session = get_session(session_id)

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Supported: images (png/jpg/webp/gif), PDF, PPTX, Mermaid/text (mmd/md/txt).",
        )

    upload_path = os.path.join(UPLOAD_DIR, f"{session_id}{file_ext}")

    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        session["image_path"] = upload_path
        log.info("Uploaded artifact for session %s: %s", session_id, upload_path)

        response = {"session_id": session_id, "filename": file.filename}
        if is_image(file_ext):
            response["image_url"] = f"/uploads/{Path(upload_path).name}"
        return response
    except Exception as e:
        log.error("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to upload file")


@app.post("/api/analyze")
async def analyze(request: ThreatModelRequest) -> ThreatModelResponse:
    """Analyze the uploaded artifact and generate a full security review.

    This is a blocking endpoint - the model will process the artifact and
    generate the review before returning.
    """
    from file_ingest import is_image

    session = get_session(request.session_id)

    if not session["image_path"] or not os.path.exists(session["image_path"]):
        raise HTTPException(status_code=400, detail="No file uploaded for this session")

    try:
        log.info("Analyzing artifact for session %s (web_search=%s)", request.session_id, request.use_web_search)

        threat_model = analyze_architecture(session["image_path"], use_web_search=request.use_web_search)
        session["threat_model"] = threat_model

        saved_path = save_threat_model(
            threat_model,
            session["image_path"],
            folder=THREAT_MODELS_DIR
        )
        session["saved_path"] = saved_path

        artifact_ext = Path(session["image_path"]).suffix.lower()
        image_url = f"/uploads/{Path(session['image_path']).name}" if is_image(artifact_ext) else ""

        return ThreatModelResponse(
            session_id=request.session_id,
            threat_model=threat_model,
            saved_path=saved_path,
            image_url=image_url,
        )

    except Exception as e:
        log.error("Analysis failed for session %s: %s", request.session_id, e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationRequest) -> dict:
    """Generate security recommendations based on threat model."""
    session = get_session(request.session_id)
    
    if not request.threat_model:
        raise HTTPException(status_code=400, detail="No threat model provided")
    
    try:
        log.info("Generating recommendations for session %s", request.session_id)
        
        recommendations = generate_recommendations(request.threat_model)
        session["recommendations"] = recommendations
        
        return {
            "session_id": request.session_id,
            "recommendations": recommendations,
        }
    
    except Exception as e:
        log.error("Recommendation generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat about the threat model.
    
    Accepts user messages and returns assistant responses based on the
    threat model and conversation history.
    """
    session = get_session(request.session_id)
    
    if not session["threat_model"]:
        raise HTTPException(status_code=400, detail="No threat model generated yet")
    
    try:
        log.info("Chat message from session %s", request.session_id)
        
        # Add user message to history
        session["chat_history"].append({
            "role": "user",
            "content": request.message,
        })
        
        # Generate response
        response = threat_model_chat(
            session["threat_model"],
            request.message,
            session["chat_history"],
        )
        
        # Add assistant response to history
        session["chat_history"].append({
            "role": "assistant",
            "content": response,
        })
        
        return ChatResponse(
            role="assistant",
            content=response,
        )
    
    except Exception as e:
        log.error("Chat failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/api/session/{session_id}")
async def get_session_data(session_id: str) -> dict:
    """Get current session data."""
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "threat_model": session["threat_model"],
        "recommendations": session["recommendations"],
        "chat_history": session["chat_history"],
        "image_path": session["image_path"],
        "saved_path": session["saved_path"],
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Clean up a session and its temporary files."""
    session = get_session(session_id)  # Verify session exists
    cleanup_session(session_id)
    return {"message": "Session deleted"}


@app.get("/api/threat_models")
async def list_threat_models() -> dict:
    """List all saved threat models."""
    try:
        models = []
        if os.path.exists(THREAT_MODELS_DIR):
            for filename in os.listdir(THREAT_MODELS_DIR):
                if filename.endswith(".md"):
                    filepath = os.path.join(THREAT_MODELS_DIR, filename)
                    models.append({
                        "filename": filename,
                        "path": filepath,
                        "url": f"/threat_models/{filename}",
                    })
        return {"threat_models": models}
    except Exception as e:
        log.error("Failed to list threat models: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list threat models")


@app.get("/threat_models/{filename}")
async def download_threat_model(filename: str) -> FileResponse:
    """Download a saved threat model file."""
    filepath = os.path.join(THREAT_MODELS_DIR, filename)
    
    # Security: prevent directory traversal
    if not os.path.abspath(filepath).startswith(os.path.abspath(THREAT_MODELS_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath, media_type="text/markdown", filename=filename)


@app.get("/uploads/{filename}")
async def get_upload(filename: str) -> FileResponse:
    """Serve uploaded images."""
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    # Security: prevent directory traversal
    if not os.path.abspath(filepath).startswith(os.path.abspath(UPLOAD_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


# ===== Error Handlers =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ===== Startup/Shutdown =====

@app.on_event("startup")
async def startup():
    """Initialize on startup. Raising here aborts the server start."""
    log.info("Sharlok starting up...")
    # Runs for every entry point (main.py, uvicorn app:app, python app.py), so Ollama is always checked.
    try:
        await asyncio.to_thread(ensure_ollama_ready)
    except Exception as e:
        log.error("Startup check failed: %s", e)
        raise
    # Load knowledge base in background (non-blocking)
    def init_kb():
        try:
            from knowledge_base import build_index_if_missing
            log.info("Checking knowledge base in background...")
            build_index_if_missing()
            log.info("Knowledge base ready")
        except Exception as e:
            log.warning("Knowledge base initialization failed: %s", e)
    
    _kb_executor.submit(init_kb)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    log.info("Sharlok shutting down...")
    # Cleanup temporary files
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
