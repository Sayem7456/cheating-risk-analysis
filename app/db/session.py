from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings

sync_engine = create_engine(
    settings.db_url_sync,
    echo=settings.debug,
    pool_size=5,
)


def get_sync_db() -> Session:
    return Session(sync_engine)
