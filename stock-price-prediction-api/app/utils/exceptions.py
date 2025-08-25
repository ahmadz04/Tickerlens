class StockPredictionError(Exception):
    """Base exception for stock prediction errors"""
    pass

class DataFetchError(StockPredictionError):
    """Raised when data fetching fails"""
    pass

class ModelNotFoundError(StockPredictionError):
    """Raised when requested model is not found"""
    pass

class InvalidTickerError(StockPredictionError):
    """Raised when ticker symbol is invalid"""
    pass

class InsufficientDataError(StockPredictionError):
    """Raised when not enough data is available for prediction"""
    pass

class ModelTrainingError(StockPredictionError):
    """Raised when model training fails"""
    pass