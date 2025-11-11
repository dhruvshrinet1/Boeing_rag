"""
Main entry point for Boeing 737 Manual RAG Service
Run this file to start the API server: python main.py
"""
import os
import sys
import uvicorn
from dotenv import load_dotenv

from logger import GLOBAL_LOGGER as log


def main():
    """
    Main function to start the Boeing RAG API server.
    """
    # Load environment variables
    if os.getenv("ENV", "local").lower() != "production":
        load_dotenv()
        log.info("Environment variables loaded from .env file")
    else:
        log.info("Running in production mode")

    # Get server configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    log.info("Starting Boeing 737 Manual RAG Service",
            host=host,
            port=port,
            reload=reload)

    # Print startup information
    print("=" * 60)
    print("Boeing 737 Manual RAG Service")
    print("=" * 60)
    print(f"Server starting at: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")
    print(f"Health check: http://{host}:{port}/health")
    print(f"Query endpoint: http://{host}:{port}/query")
    print("=" * 60)
    print()

    try:
        # Start the server
        uvicorn.run(
            "src.api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

    except KeyboardInterrupt:
        log.info("Server stopped by user")
        print("\nServer stopped.")

    except Exception as e:
        log.error("Failed to start server", error=str(e))
        print(f"\nError starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
