import json
from datetime import datetime
from decimal import Decimal
from enum import Enum

import numpy as np
from uuid import UUID

from pydantic import BaseModel


class RedisEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, Enum):
            return obj.value
        elif obj is None:
            return 'None'
        return super(RedisEncoder, self).default(obj)
