from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class Conversation(Base):
    """Model for storing conversation sessions"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class Message(Base):
    """Model for storing individual messages in conversations"""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    risk_level = Column(String(20), nullable=True)
    confidence = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)

def get_engine():
    """Get database engine using environment variables"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return create_engine(database_url)

def init_db():
    """Initialize database tables"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine

def get_session():
    """Get a database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()
