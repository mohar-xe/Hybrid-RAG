"""Typed exception hierarchy.

Every project exception derives from ``BaseError`` so callers can catch the
whole family with a single ``except BaseError`` when desired, while still being
able to target a specific subclass.
"""

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
class ExtractionError(BaseError):
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

class EmbeddingError(BaseError):
    pass


class DatabaseError(BaseError):
    pass

"""
+------------------------------+
|             Graph            |
+------------------------------+
"""

class GraphError(BaseError):
    pass
