import sys
import logging

logger = logging.getLogger(__name__)

def exit_from_program():
    logger.info("Program has shutted down...")
    sys.exit()