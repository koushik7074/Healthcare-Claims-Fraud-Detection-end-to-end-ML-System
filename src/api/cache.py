import redis
import json
import hashlib
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

def make_cache_key(payload: dict) -> str:
    """
    Creating a deterministic cache key from request payload
    """
    payload_str = json.dumps(payload, sort_keys=True)
    return f"fraud_pred:{hashlib.md5(payload_str.encode()).hexdigest()}"
