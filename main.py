import logging
from colorama import Fore, Style

from create_ayah_chunks import create_ayah_chunks
from embed_and_store import embed_and_store

logging.basicConfig(
    level=logging.INFO,
    format=f'{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.GREEN}%(levelname)s{Style.RESET_ALL} - %(message)s'
)
logger = logging.getLogger(__name__)

data_path = "data/quran.csv"

chunks = create_ayah_chunks(data_path)

stats = embed_and_store(chunks, "sakinah-app")

logger.info(f"{Fore.GREEN}First chunk: {chunks[0]}{Style.RESET_ALL}")
logger.info(f"{Fore.GREEN}Second chunk: {chunks[1]}{Style.RESET_ALL}")
logger.info(f"{Fore.GREEN}Last chunk: {chunks[-1]}{Style.RESET_ALL}")