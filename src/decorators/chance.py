from functools import wraps
from random import random


def chance(probability: float):
    """
    Decorator that makes a handler execute only with a certain probability.

    Args:
        probability (float): Probability of execution (0.0 to 1.0)

    Returns:
        The decorated handler function
    """

    def decorator(handler):
        @wraps(handler)
        async def wrapper(*args, **kwargs):
            if random() < probability:
                return await handler(*args, **kwargs)
            return None

        return wrapper

    return decorator
