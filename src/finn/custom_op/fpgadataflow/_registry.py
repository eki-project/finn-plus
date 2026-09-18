# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Shared custom-op registry used by the three fpgadataflow ONNX domains.

``qonnx.custom_op.registry`` resolves a node's ``domain`` by importing the module
of that name and reading its ``custom_op`` dictionary. Each domain package
therefore owns one such dictionary plus a decorator that fills it; this module
builds both so the three packages do not each re-implement them.
"""

from collections.abc import Callable
from qonnx.custom_op.base import CustomOp
from typing import TypeVar

from finn.util.exception import FINNInternalError

_OpT = TypeVar("_OpT", bound=CustomOp)


def make_registry(
    base_cls: type[CustomOp],
) -> tuple[dict[str, type[CustomOp]], Callable[[type[_OpT]], type[_OpT]]]:
    """Create a custom-op dictionary and the decorator that registers into it.

    Args:
        base_cls: Class every registered op must subclass. This is what
            distinguishes the domains from one another: ``CustomOp`` for the
            abstract layers, ``HLSBackend`` / ``RTLBackend`` for the backends.

    Returns:
        A ``(custom_op, register_custom_op)`` pair. ``custom_op`` maps class name
        to class and is the dictionary qonnx reads; ``register_custom_op`` is a
        pass-through decorator that inserts a class into it.

    """
    custom_op: dict[str, type[CustomOp]] = {}

    def register_custom_op(cls: type[_OpT]) -> type[_OpT]:
        """Register ``cls`` into the ``custom_op`` dictionary by its name."""
        if not issubclass(cls, base_cls):
            raise FINNInternalError(f"{cls} must subclass {base_cls}")
        custom_op[cls.__name__] = cls
        # Pass through the class unmodified so this can be used as a decorator
        return cls

    return custom_op, register_custom_op
