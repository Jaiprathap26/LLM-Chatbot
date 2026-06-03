import logging
from memory.long_term_memory import Message, LongTermMemory

logger = logging.getLogger(__name__)


class BufferMemory:
    def __init__(self, long_term_memory: LongTermMemory):
        self.ltm = long_term_memory

    def get_recent_messages(
            self,
            session_id: str,
            limit: int = 10) -> list[dict]:
        """
        Retrieves the last N messages for a specific session ID directly from the database.
        Returns them in chronological order.
        """
        db_session = self.ltm.SessionLocal()
        try:
            results = db_session.query(Message).filter(
                Message.session_id == session_id
            ).order_by(Message.created_at.desc()).limit(limit).all()

            # Reverse to get chronological order (oldest to newest)
            results.reverse()

            return [{"role": msg.role, "content": msg.content}
                    for msg in results]
        except Exception as e:
            logger.error(f"Failed to retrieve buffer memory: {e}")
            return []
        finally:
            db_session.close()
