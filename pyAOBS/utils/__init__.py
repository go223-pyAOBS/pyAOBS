"""
Utility functions for the seismic_processing package.
"""

import logging

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't already have handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set level to INFO if not set
        if not logger.level:
            logger.setLevel(logging.INFO)
    
    return logger

# Import utility functions here
__all__ = []  # Add utility functions when created 