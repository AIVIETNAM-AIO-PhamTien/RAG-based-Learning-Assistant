from app.db.models import ChatMessage, ChatSession, ChatSessionDocument, Chunk, Document, Flashcard


def test_phase_1_tables_are_registered() -> None:
    table_names = {
        Document.__tablename__,
        Chunk.__tablename__,
        ChatSession.__tablename__,
        ChatSessionDocument.__tablename__,
        ChatMessage.__tablename__,
        Flashcard.__tablename__,
    }

    assert table_names == {
        "documents",
        "chunks",
        "chat_sessions",
        "chat_session_documents",
        "chat_messages",
        "flashcards",
    }
    assert Chunk.__table__.c.embedding.type.dim == 384
