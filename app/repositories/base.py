from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(ABC, Generic[ModelT]):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, entity: ModelT) -> ModelT:
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def get(self, id) -> ModelT | None:
        return await self.session.get(self.entity_class, id)

    async def update(self, entity: ModelT) -> ModelT:
        await self.session.merge(entity)
        await self.session.flush()
        return entity

    async def delete(self, entity: ModelT) -> None:
        await self.session.delete(entity)
        await self.session.flush()

    @property
    @abstractmethod
    def entity_class(self) -> type[ModelT]:
        ...
