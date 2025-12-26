# memory/user_profile_store.py
import sys
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.schemas import UserProfileSchema 
from config.settings import settings
from utils.json_utils import safe_json_load
from utils.logger import get_logger

try:
    from retrieval.pinecone_client import index
except ImportError:
    index = None

logger = get_logger("USER_PROFILE")

class UserProfileStore:
    """
    Manages persistence of the UserProfileSchema in Pinecone.
    """
    PROFILE_NAMESPACE = "user_profiles"
    
    def __init__(self):
        if not index:
            logger.warning("UserProfileStore: Pinecone unavailable (Local Mode).")

    def get_profile(self, user_id: str) -> UserProfileSchema:
        """Fetch profile or return default."""
        if not index:
            return UserProfileSchema(user_id=user_id)

        try:
            result = index.fetch(ids=[user_id], namespace=self.PROFILE_NAMESPACE)
            if result and user_id in result['vectors']:
                meta = result['vectors'][user_id].get('metadata', {})
                data = safe_json_load(meta.get('profile_data'))
                if data:
                    return UserProfileSchema.model_validate(data)
            
            return UserProfileSchema(user_id=user_id)
        except Exception as e:
            logger.error(f"Fetch profile error: {e}")
            return UserProfileSchema(user_id=user_id)

    def update_profile(self, profile: UserProfileSchema):
        """Force write to DB."""
        if not index: return
        try:
            # Create a placeholder vector. 
            # CRITICAL FIX: Pinecone rejects vectors that are all zeros.
            # We set the first element to 1.0 to ensure it's a valid vector.
            placeholder = [0.0] * settings.EMBEDDING_DIMENSION
            if placeholder:
                placeholder[0] = 1.0

            index.upsert(
                vectors=[{
                    'id': profile.user_id,
                    'values': placeholder,
                    'metadata': {
                        'profile_data': profile.model_dump_json(), 
                        'type': 'user_profile'
                    }
                }],
                namespace=self.PROFILE_NAMESPACE
            )
        except Exception as e:
            logger.error(f"Update profile error: {e}")

    def sync_if_changed(self, old_profile: UserProfileSchema, current_profile: UserProfileSchema):
        """
        Smart Sync: Only writes if data actually changed.
        Crucial for 'per-message' updates to avoid API limits.
        """
        if old_profile.model_dump() != current_profile.model_dump():
            logger.info(f"Syncing profile changes for {current_profile.user_id}...")
            self.update_profile(current_profile)