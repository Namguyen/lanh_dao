import os
import logging
from typing import Final

import dotenv

dotenv.load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)

# Elasticsearch configuration
ES_HOST: Final[str] = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER: Final[str] = os.getenv("ES_USER", "elastic")
ES_PASS: Final[str] = os.getenv("ES_PASS", "")
ES_INDEX: Final[str] = os.getenv("ES_INDEX", "lanh_dao")


# DeepSeek LLM configuration
DEEPSEEK_API_KEY: Final[str] = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: Final[str] = "https://api.deepseek.com"
DEEPSEEK_MODEL: Final[str] = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MODEL_FAST: Final[str] = os.getenv("DEEPSEEK_MODEL_FAST", DEEPSEEK_MODEL)


# Web search (Tavily) configuration
TAVILY_API_KEY: Final[str] = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL: Final[str] = "https://api.tavily.com/search"
TAVILY_TIMEOUT_SECONDS: Final[int] = 10
TAVILY_MAX_RESULTS: Final[int] = 3
TAVILY_NEWS_DAYS: Final[int] = 30


def validate_config() -> bool:
    errors = []

    if not DEEPSEEK_API_KEY:
        errors.append("DEEPSEEK_API_KEY is required but not set")

    if not ES_PASS:
        log.warning("ES_PASS is empty (insecure for production)")

    if not TAVILY_API_KEY:
        log.warning("TAVILY_API_KEY is not set (internet search disabled)")

    if errors:
        for error in errors:
            log.error(error)
        return False

    return True

