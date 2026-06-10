import uvicorn

from src.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run("src.interface.app:app", host=settings.app_host, port=settings.app_port, reload=True)


if __name__ == "__main__":
    main()
