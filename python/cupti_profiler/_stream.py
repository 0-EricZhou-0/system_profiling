"""Thin RAII wrapper around create_cuda_stream / destroy_cuda_stream."""

from ._native import create_cuda_stream, destroy_cuda_stream


class CudaStream:
    """Context manager that yields a cudaStream_t handle (as int)."""

    handle: int

    def __init__(self) -> None:
        self.handle = 0

    def __enter__(self) -> int:
        self.handle = create_cuda_stream()
        return self.handle

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.handle:
            destroy_cuda_stream(self.handle)
            self.handle = 0
        return False
