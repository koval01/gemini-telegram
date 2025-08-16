import asyncio
import contextlib
import io
import logging
import math
import time
import random
import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

from RestrictedPython import compile_restricted, safe_builtins, PrintCollector
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import guarded_unpack_sequence, guarded_iter_unpack_sequence
from sklearn import linear_model, metrics, preprocessing, model_selection

try:
    from RestrictedPython.RestrictingNodeTransformer import RestrictingNodeTransformer
except ImportError:
    try:
        from RestrictedPython.transformer import RestrictingNodeTransformer
    except ImportError as e:
        raise ImportError(
            "Could not import RestrictingNodeTransformer. "
            "Install/upgrade RestrictedPython (e.g. pip install -U 'RestrictedPython>=6.1')."
        ) from e

logger = logging.getLogger(__name__)


class AllowedModules:
    """Centralized configuration for allowed modules and their components."""

    BASE_MODULES = {
        'math': math,
        'time': time,
        'random': random,
        'datetime': datetime
    }

    SCIENTIFIC_MODULES = {
        'numpy': np,
        'np': np,
        'pandas': pd,
        'pd': pd,
        'sklearn': {
            'linear_model': linear_model,
            'metrics': metrics,
            'preprocessing': preprocessing,
            'model_selection': model_selection
        },
        'sklearn.linear_model': linear_model,
        'sklearn.metrics': metrics,
        'sklearn.preprocessing': preprocessing,
        'sklearn.model_selection': model_selection
    }

    @classmethod
    def get_allowed_modules(cls) -> Dict[str, Any]:
        """Combine all allowed modules into a single dictionary."""
        allowed = {}
        allowed.update(cls.BASE_MODULES)
        allowed.update(cls.SCIENTIFIC_MODULES)
        return allowed

    @classmethod
    def get_import_whitelist(cls) -> Dict[str, bool]:
        """Generate a whitelist of allowed import statements."""
        whitelist = {}
        for module in cls.BASE_MODULES:
            whitelist[f'import {module}'] = True
            whitelist[f'from {module} import'] = True
        for module in cls.SCIENTIFIC_MODULES:
            if not isinstance(cls.SCIENTIFIC_MODULES[module], dict):
                whitelist[f'import {module}'] = True
                whitelist[f'from {module} import'] = True
                whitelist[f'import {module} as'] = True
        return whitelist


def safe_import(name, *_args, **_kwargs):
    """Handler for import statements in restricted code."""
    allowed_modules = AllowedModules.get_allowed_modules()
    if name in allowed_modules:
        return allowed_modules[name]
    raise ImportError(f"Import of '{name}' is not allowed")


def _safe_getattr(obj, attr):
    """Safe attribute access handler."""
    forbidden_attrs = {'__class__', '__globals__', '__code__', '__builtins__'}
    if attr in forbidden_attrs:
        raise AttributeError(f"Access to '{attr}' is not allowed")
    return getattr(obj, attr)


def _inplacevar_wrapper(op, target, arg):
    """Handle in-place operations safely."""
    if op == '+=':
        return target + arg
    elif op == '-=':
        return target - arg
    elif op == '*=':
        return target * arg
    elif op == '/=':
        return target / arg
    elif op == '//=':
        return target // arg
    elif op == '%=':
        return target % arg
    elif op == '**=':
        return target ** arg
    elif op == '<<=':
        return target << arg
    elif op == '>>=':
        return target >> arg
    elif op == '&=':
        return target & arg
    elif op == '^=':
        return target ^ arg
    elif op == '|=':
        return target | arg
    else:
        raise ValueError(f"Unsupported in-place operation: {op}")


class DunderFriendlyPolicy(RestrictingNodeTransformer):
    """
    Policy that allows a minimal, explicit set of dunder names and the
    single-underscore loop variable '_' while keeping other protections intact.
    """
    ALLOWED_DUNDERS = {
        "__init__", "__add__", "__mul__", "__str__", "__repr__",
        "__radd__", "__rmul__", "__iadd__", "__imul__", "__metaclass__"
    }

    def check_name(self, node, name, **kwargs):
        if name in self.ALLOWED_DUNDERS or name == "_":
            return None
        try:
            return super().check_name(node, name, **kwargs)
        except TypeError:
            return super().check_name(node, name)


def get_safe_globals() -> Dict[str, Any]:
    """Generate the safe globals dictionary with all allowed functionality."""
    base_builtins = {
        **safe_builtins,
        'None': None,
        'bool': bool,
        'int': int,
        'float': float,
        'str': str,
        'tuple': tuple,
        'list': list,
        'dict': dict,
        'set': set,
        'frozenset': frozenset,
        'len': len,
        'range': range,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'divmod': divmod,
        'enumerate': enumerate,
        'map': map,
        '__import__': safe_import,
        'print': print,
        '_getattr_': _safe_getattr,
        'ValueError': ValueError,
    }

    return {
        '__builtins__': base_builtins,
        '_getiter_': default_guarded_getiter,
        '_unpack_sequence_': guarded_unpack_sequence,
        '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
        '_getitem_': lambda x, y: x[y],
        '_write_': lambda x: x,
        '_inplacevar_': _inplacevar_wrapper,
        '_print_': PrintCollector,
        '__name__': '__main__',
        '__metaclass__': type,
        **AllowedModules.get_allowed_modules(),
        '__add__': lambda a, b: a + b,
        '__sub__': lambda a, b: a - b,
        '__mul__': lambda a, b: a * b,
        '__truediv__': lambda a, b: a / b,
        '__floordiv__': lambda a, b: a // b,
        '__mod__': lambda a, b: a % b,
        '__pow__': lambda a, b: a ** b,
        '__eq__': lambda a, b: a == b,
        '__ne__': lambda a, b: a != b,
        '__lt__': lambda a, b: a < b,
        '__le__': lambda a, b: a <= b,
        '__gt__': lambda a, b: a > b,
        '__ge__': lambda a, b: a >= b,
    }


class RestrictedExecutor:
    """
    Handles async execution of restricted Python code with math, random numpy, and sklearn support.
    """

    DANGEROUS_PATTERNS = [
        'open(', 'exec(', 'eval(', 'subprocess(', 'exit(',
        'os.', 'sys.', 'shutil.', 'glob.', 'pickle.',
        'socket.', 'ctypes.', 'codecs.', 'builtins.',
        'importlib.', 'compile(', 'input(', 'help('
    ]

    @classmethod
    async def execute_code_async(cls, source_code: str, timeout: float = 3.0) -> str:
        if not source_code.strip():
            return "Empty code provided"

        source_code = cls._clean_imports(source_code)

        if cls._contains_dangerous_code(source_code):
            return "Error: Dangerous code patterns detected"

        try:
            byte_code = compile_restricted(
                source_code,
                '<inline>',
                'exec',
                policy=DunderFriendlyPolicy
            )

            def _execute():
                exec_locals = {}
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf):
                        exec(byte_code, get_safe_globals(), exec_locals)
                        for name, value in exec_locals.items():
                            if callable(value):
                                get_safe_globals()[name] = value
                        if 'main' in exec_locals and callable(exec_locals['main']):
                            _result = exec_locals['main']()
                            if hasattr(_result, 'tolist'):
                                return str(_result.tolist())
                            if _result is not None:
                                return str(_result)
                    out = buf.getvalue().strip()
                    return out if out else "Code executed"
                except Exception as exception:
                    return f"Error: {str(exception)}"

            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _execute),
                timeout=timeout
            )
            return result

        except asyncio.TimeoutError:
            return "Error: Execution timed out"
        except Exception as _exc:
            error_msg = f"Execution error: {str(_exc)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @classmethod
    def _contains_dangerous_code(cls, source_code: str) -> bool:
        """Check for dangerous code patterns."""
        lower_code = source_code.lower()

        if any(pattern in lower_code for pattern in cls.DANGEROUS_PATTERNS):
            return True

        import_whitelist = AllowedModules.get_import_whitelist()
        for line in source_code.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith(('import ', 'from ')):
                if not any(stripped.startswith(allowed) for allowed in import_whitelist):
                    return True
        return False

    @staticmethod
    def format_code_response(code: str, result: str) -> str:
        """
        Format code and result into a markdown response with math support.
        """

        def escape_md(text: str) -> str:
            return text.translate(str.maketrans({
                "_": r"\_",
                "*": r"\*",
                "[": r"\[",
                "]": r"\]",
                "(": r"\(",
                ")": r"\)",
                "~": r"\~",
                "`": r"\`",
                ">": r"\>",
                "#": r"\#",
                "+": r"\+",
                "-": r"\-",
                "=": r"\=",
                "|": r"\|",
                "{": r"\{",
                "}": r"\}",
                ".": r"\.",
                "!": r"\!"
            }))

        escaped_code = escape_md(code)
        escaped_result = escape_md(result)

        if any(op in result for op in ['=', '≈', '≠', '<', '>', '≤', '≥', '±', '∑', '∏', '∫']):
            return (
                f"```python\n{escaped_code}\n```\n"
                f"\n\nResult:"
                f"```\n{escaped_result}\n```\n"
                f"\n\nFormatted math:\n"
                f"`{escaped_result.replace('=', ' = ').replace('*', ' × ')}`"
            )
        return (
            f"```python\n{escaped_code}\n```\n"
            f"\n\nResult:"
            f"```\n{escaped_result}\n```"
        )

    @staticmethod
    def _clean_imports(code: str) -> str:
        """Clean imports while allowing only whitelisted imports."""
        import_whitelist = AllowedModules.get_import_whitelist()
        lines = code.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip().lower()
            if stripped.startswith(('import ', 'from ')):
                if any(stripped.startswith(allowed) for allowed in import_whitelist):
                    cleaned_lines.append(line)
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()
