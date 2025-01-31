from abc import ABC, abstractmethod


class ResumableIterator(ABC):
    @abstractmethod
    def has_next(self) -> bool:
        pass

    @abstractmethod
    def next(self):
        pass


class Resumable2DIterator(ResumableIterator):
    def __init__(self, matrix):
        self.matrix = matrix
        self.row = 0
        self.col = 0
        self._advance_next()

    def _advance_next(self):
        while self.row < len(self.matrix) and self.col >= len(self.matrix[self.row]):
            self.row += 1
            self.col = 0

    def has_next(self) -> bool:
        return self.row < len(self.matrix)

    def next(self):
        if not self.has_next():
            raise StopIteration("No more elements.")

        value = self.matrix[self.row][self.col]
        self.col += 1
        self._advance_next()
        return value


def test_resumable_2d_iterator():
    matrix = [
        [1, 2, 3],
        [],
        [4, 5],
        [],
        [6]
    ]

    iterator = Resumable2DIterator(matrix)
    result = []
    while iterator.has_next():
        result.append(iterator.next())

    print(result)


test_resumable_2d_iterator()
