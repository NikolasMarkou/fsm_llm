import os
import sys
from loguru import logger
from functools import wraps

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)


# Define a filter to handle logs without conversation_id
def prepare_log_record(record):
    if "conversation_id" not in record["extra"]:
        record["extra"]["conversation_id"] = "GENERAL"
    return record


# Remove default handler
logger.remove()

# Add console handler with colors and conversation_id
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<yellow>conv_id: {extra[conversation_id]:<12}</yellow> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "<level>{message}</level>",
    level="INFO",
    filter=prepare_log_record
)

# Add file handler with rotation and conversation_id
logger.add(
    "logs/neural-fsm_{time}.log",
    rotation="10 MB",  # Rotate when file reaches 10MB
    retention="1 month",  # Keep logs for 1 month
    compression="zip",  # Compress rotated logs
    format="{time:YYYY-MM-DD HH:mm:ss} | "
           "{level: <8} | "
           "conv_id: {extra[conversation_id]:<12} | "
           "{name}:{function}:{line} | "
           "{message}",
    level="DEBUG",
    filter=prepare_log_record
)


def with_conversation_context(func):
    @wraps(func)  # Preserve function metadata
    def wrapper(self, conversation_id, *args, **kwargs):
        # Bind the conversation ID to the logger
        log = logger.bind(conversation_id=conversation_id)

        # Call the original function with the contextual logger
        return func(self, conversation_id, *args, log=log, **kwargs)

    return wrapper