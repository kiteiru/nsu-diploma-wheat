import logging

logger = logging.getLogger(__name__)

def log_metrics(mean_precision, mean_recall, mean_fscore):
    print("log_metrics func")
    logger.info(f"Precision: {mean_precision}")
    logger.info(f"Recall: {mean_recall}")
    logger.info(f"F-score: {mean_fscore}")