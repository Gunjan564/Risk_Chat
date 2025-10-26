from db_models import PostAnalysisLog, get_session, init_db # Import simplified model
from datetime import datetime
import uuid
from sqlalchemy.exc import SQLAlchemyError

def initialize_database():
    """Initialize database and create tables if they don't exist"""
    try:
        init_db()
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False
def log_post_analysis(content, risk_level, confidence):
    """
    Log a single post analysis result to the database.
    """
    try:
        session = get_session()
        
        # Create a new log entry
        log_entry = PostAnalysisLog(
            content=content,
            risk_level=risk_level,
            confidence=confidence,
            timestamp=datetime.now(),
            source='streamlit_web'
        )
        
        session.add(log_entry)
        session.commit()
        session.close()
        return True
    except SQLAlchemyError as e:
        print(f"Error logging analysis to database: {e}")
        # Rollback in case of error
        if 'session' in locals() and not session.closed:
            session.rollback()
            session.close()
        return False
    except Exception as e:
        print(f"Unexpected error in log_post_analysis: {e}")
        return False
# def generate_session_id():
#     """Generate a unique session ID"""
#     return str(uuid.uuid4())

# def save_message(session_id, role, content, risk_level=None, confidence=None):
#     """Save a message to the database"""
#     try:
#         session = get_session()
        
#         # Create or update conversation
#         conversation = session.query(Conversation).filter_by(session_id=session_id).first()
#         if not conversation:
#             conversation = Conversation(session_id=session_id)
#             session.add(conversation)
#         else:
#             conversation.updated_at = datetime.now()
        
#         # Create message
#         message = Message(
#             session_id=session_id,
#             role=role,
#             content=content,
#             risk_level=risk_level,
#             confidence=confidence,
#             timestamp=datetime.now()
#         )
#         session.add(message)
#         session.commit()
#         session.close()
#         return True
#     except Exception as e:
#         print(f"Error saving message: {e}")
#         return False

# def load_conversation(session_id):
#     """Load all messages for a given session"""
#     try:
#         session = get_session()
#         messages = session.query(Message).filter_by(session_id=session_id).order_by(Message.timestamp).all()
        
#         result = []
#         for msg in messages:
#             message_data = {
#                 'role': msg.role,
#                 'content': msg.content,
#                 'timestamp': msg.timestamp
#             }
#             if msg.risk_level:
#                 message_data['risk_level'] = msg.risk_level
#             if msg.confidence:
#                 message_data['confidence'] = msg.confidence
#             result.append(message_data)
        
#         session.close()
#         return result
#     except Exception as e:
#         print(f"Error loading conversation: {e}")
#         return []

# def get_all_sessions():
#     """Get all conversation sessions"""
#     try:
#         session = get_session()
#         conversations = session.query(Conversation).order_by(Conversation.updated_at.desc()).all()
        
#         result = []
#         for conv in conversations:
#             result.append({
#                 'session_id': conv.session_id,
#                 'created_at': conv.created_at,
#                 'updated_at': conv.updated_at
#             })
        
#         session.close()
#         return result
#     except Exception as e:
#         print(f"Error getting sessions: {e}")
#         return []

# def delete_conversation(session_id):
#     """Delete a conversation and all its messages"""
#     try:
#         session = get_session()
#         session.query(Message).filter_by(session_id=session_id).delete()
#         session.query(Conversation).filter_by(session_id=session_id).delete()
#         session.commit()
#         session.close()
#         return True
#     except Exception as e:
#         print(f"Error deleting conversation: {e}")
#         return False

# def get_conversation_stats(session_id):
#     """Get statistics for a conversation"""
#     try:
#         session = get_session()
#         messages = session.query(Message).filter_by(session_id=session_id, role='user').all()
        
#         if not messages:
#             session.close()
#             return None
        
#         risk_levels = {}
#         total_confidence = 0
#         count = 0
        
#         for msg in messages:
#             if msg.risk_level:
#                 risk_levels[msg.risk_level] = risk_levels.get(msg.risk_level, 0) + 1
#             if msg.confidence:
#                 total_confidence += msg.confidence
#                 count += 1
        
#         avg_confidence = total_confidence / count if count > 0 else 0
        
#         session.close()
#         return {
#             'total_messages': len(messages),
#             'risk_distribution': risk_levels,
#             'average_confidence': avg_confidence
#         }
#     except Exception as e:
#         print(f"Error getting conversation stats: {e}")
#         return None
