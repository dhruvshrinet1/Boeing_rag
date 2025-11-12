import os
import sys
import uvicorn
from dotenv import load_dotenv
from logger import GLOBAL_LOGGER as log


def main():
    if os.getenv("ENV", "local").lower() != "production":
        load_dotenv()
        log.info("Loaded .env file")

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    log.info("Starting service", host=host, port=port)

    print("=" * 60)
    print("Boeing 737 Manual RAG Service")
    print("=" * 60)
    print(f"Server: http://{host}:{port}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"Health: http://{host}:{port}/health")
    print("=" * 60)
    print()

    try:
        uvicorn.run(
            "src.api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        log.info("Stopped by user")
        print("\nServer stopped.")
    except Exception as e:
        log.error("Failed to start", error=str(e))
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
