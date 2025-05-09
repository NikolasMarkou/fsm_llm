import os
import sys
from loguru import logger

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Remove default handler
logger.remove()

# Add console handler with colors
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file handler with rotation
logger.add(
    "logs/neural-fsm_{time}.log",
    rotation="10 MB",  # Rotate when file reaches 10MB
    retention="1 month",  # Keep logs for 1 month
    compression="zip",  # Compress rotated logs
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"  # Log everything to file
)