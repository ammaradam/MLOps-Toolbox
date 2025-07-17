"""Simple task and pipeline decorators."""

from __future__ import annotations

import functools
from typing import Callable, Dict


class Task:
    """Represents a single task."""

    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f"Running task: {self.func.__name__}")
        return self.func(*args, **kwargs)


class Pipeline:
    """Represents a pipeline of tasks."""

    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f"Running pipeline: {self.func.__name__}")
        return self.func(*args, **kwargs)


TASK_REGISTRY: Dict[str, Task] = {}
PIPELINE_REGISTRY: Dict[str, Pipeline] = {}


def task(func: Callable) -> Task:
    """Decorator to register a function as a :class:`Task`."""
    t = Task(func)
    TASK_REGISTRY[func.__name__] = t
    return t


def pipeline(func: Callable) -> Pipeline:
    """Decorator to register a function as a :class:`Pipeline`."""
    p = Pipeline(func)
    PIPELINE_REGISTRY[func.__name__] = p
    return p
