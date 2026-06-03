import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime

logger = logging.getLogger(__name__)

Base = declarative_base()


class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    # text-embedding-004 is 768 dims
    embedding = Column(Vector(768), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class LongTermMemory:
    def __init__(self, db_url: str = None):
        """Initializes the connection to the database and ensures tables exist."""
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL is not set.")

        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine)
        self._initialize_db()

    def _initialize_db(self):
        """Creates the vector extension and the necessary tables if they don't exist."""
        try:
            with self.engine.connect() as conn:
                # pgvector extension needs to be created before table creation
                import sqlalchemy
                conn.execute(
                    sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def save_message(
            self,
            session_id: str,
            role: str,
            content: str,
            embedding: list[float] = None) -> None:
        """Saves a message to the database."""
        db_session = self.SessionLocal()
        try:
            msg = Message(
                session_id=session_id,
                role=role,
                content=content,
                embedding=embedding
            )
            db_session.add(msg)
            db_session.commit()
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            db_session.rollback()
        finally:
            db_session.close()

    def get_relevant_memories(
            self,
            session_id: str,
            query_embedding: list[float],
            limit: int = 3,
            exclude_last_n: int = 10) -> list[str]:
        """
        Retrieves the top-k most semantically similar messages from the database
        for a specific session, excluding the most recent N messages (buffer).
        """
        db_session = self.SessionLocal()
        try:
            # First, find the IDs of the last N messages to exclude them
            recent_msgs = db_session.query(Message.id).filter(
                Message.session_id == session_id
            ).order_by(Message.created_at.desc()).limit(exclude_last_n).all()

            recent_ids = [msg.id for msg in recent_msgs]

            # Now perform vector search, excluding those recent IDs
            query = db_session.query(Message).filter(
                Message.session_id == session_id,
                Message.embedding.is_not(None)
            )

            if recent_ids:
                query = query.filter(Message.id.not_in(recent_ids))

            # Order by cosine distance (<=>)
            results = query.order_by(
                Message.embedding.cosine_distance(query_embedding)
            ).limit(limit).all()

            return [f"{msg.role}: {msg.content}" for msg in results]
        except Exception as e:
            logger.error(f"Failed to retrieve relevant memories: {e}")
            return []
        finally:
            db_session.close()
