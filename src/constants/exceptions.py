"""
+------------------------------+
|            Global            |
+------------------------------+
"""

class BaseError(Exception):
    pass

class ConfigurationError(BaseError):
    pass

class ValidationError(BaseError):
    pass

class APIError(BaseError):
    pass

class ModelError(BaseError):
    pass

"""
+------------------------------+
|         Ingestion            |
+------------------------------+
"""
class ExtractionError(Exception):
    pass

class TextExtractionError(ExtractionError):
    pass

class AudioExtractionError(ExtractionError):
    pass

class YTSubtitleExtractionError(ExtractionError):
    pass

"""
+------------------------------+
|           Embedding          |
+------------------------------+
"""

class EmbeddingError(Exception):
    pass


class DatabaseError(Exception):
    pass