# C:\Users\hp\OneDrive\Desktop\Risk_Chat\db_models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import uuid

Base = declarative_base()

class PostAnalysisLog(Base):
    """Model for logging every single post analysis performed."""
    __tablename__ = 'post_analysis_logs'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(String(100), nullable=False, default=lambda: str(uuid.uuid4()), unique=True)
    content = Column(Text, nullable=False)
    risk_level = Column(String(20), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    source = Column(String(50), default='streamlit_web')
    
# --- The Conversation class and Message class MUST be removed for Option 2 ---
# (They were left at the end of your provided code; they are removed here)

def get_engine():
    """Get database engine using environment variables"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        # NOTE: Set DATABASE_URL environment variable (e.g., 'sqlite:///./risk_analysis_log.db')
        raise ValueError("DATABASE_URL environment variable not set")
    return create_engine(database_url)

def init_db():
    """Initialize database tables"""
    engine = get_engine()
    # This will only create the 'post_analysis_logs' table now
    Base.metadata.create_all(engine) 
    return engine

def get_session():
    """Get a database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()