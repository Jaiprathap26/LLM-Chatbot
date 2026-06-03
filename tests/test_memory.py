import pytest
from unittest.mock import MagicMock, patch
from memory.long_term_memory import LongTermMemory, Message
from memory.buffer_memory import BufferMemory


@pytest.fixture
def mock_db_session():
    with patch('memory.long_term_memory.create_engine'), \
            patch('memory.long_term_memory.sessionmaker') as mock_sessionmaker, \
            patch('memory.long_term_memory.Base.metadata.create_all'), \
            patch('memory.long_term_memory.LongTermMemory._initialize_db'):

        mock_session_class = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_class.return_value = mock_session_instance
        mock_sessionmaker.return_value = mock_session_class

        # We also need to mock os.getenv to avoid "DATABASE_URL is not set"
        with patch('os.getenv', return_value="postgresql://dummy"):
            ltm = LongTermMemory()
            yield ltm, mock_session_instance


def test_save_message(mock_db_session):
    ltm, mock_session = mock_db_session
    ltm.save_message("session_1", "user", "hello", [0.1, 0.2])

    mock_session.add.assert_called_once()
    args, _ = mock_session.add.call_args
    msg = args[0]
    assert msg.session_id == "session_1"
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.embedding == [0.1, 0.2]
    mock_session.commit.assert_called_once()


def test_buffer_memory_get_recent(mock_db_session):
    ltm, mock_session = mock_db_session

    # Mock the query chain
    mock_query = mock_session.query.return_value
    mock_filter = mock_query.filter.return_value
    mock_order = mock_filter.order_by.return_value
    mock_limit = mock_order.limit.return_value

    # Return some mock messages
    msg1 = Message(session_id="session_1", role="user", content="hello")
    msg2 = Message(session_id="session_1", role="assistant", content="hi")
    # DB returns newest first typically if desc()
    mock_limit.all.return_value = [msg2, msg1]

    buffer = BufferMemory(ltm)
    results = buffer.get_recent_messages("session_1", limit=2)

    assert len(results) == 2
    # Should be reversed to chronological
    assert results[0]["role"] == "user"
    assert results[1]["role"] == "assistant"
