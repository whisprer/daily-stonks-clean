from __future__ import annotations
from app.db import engine, Base
from app import models  # noqa: F401

def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("DB tables created.")

if __name__ == "__main__":
    main()
