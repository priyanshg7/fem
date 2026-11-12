"""
Run script for launching the SimuStruct FastAPI backend server.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn
from backend.app.core.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.PROJECT_NAME} v{settings.API_VERSION} on {settings.HOST}:{settings.PORT}")
    print(f"Interactive OpenAPI docs available at: http://localhost:{settings.PORT}/docs")
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1
    )
