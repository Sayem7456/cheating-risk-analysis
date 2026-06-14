from abc import ABC, abstractmethod
from typing import Any


class BaseAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, data: Any) -> dict[str, Any]:
        ...
