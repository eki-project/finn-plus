from __future__ import annotations

import json
import logging
import socket
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger("finn_logger")


class RunStatus(str, Enum):
    """Status of a FINN run."""

    RUNNING = "running"
    FAIL = "fail"
    SUCCESS = "success"
    UNKNOWN = "unknown"


class MessageType(str, Enum):
    """All types of messages that can be sent to the status server."""

    STATUS_UPDATE = "status_update"
    CURRENT_STEP_UPDATE = "current_step_update"


STATUS_SERVER_MESSAGE_LENGTH_BYTES = 8


# Server status updates
class _StatusServer:
    """Manages the connection to the status server."""

    def __init__(self) -> None:
        """Create a new status serverconnection at the given path."""
        self._status_server: socket.socket | None = None

    def connected(self) -> bool:
        """Return whether we are connected to a status server."""
        return self._status_server is not None

    def initialize_status_socket(self, socket_path: Path) -> bool:
        """Initialize the socket to the given path. If successful return True, False otherwise."""
        if not socket_path.exists():
            return False
        self._status_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._status_server.connect(str(socket_path))
        return True

    def _send_message(self, msg: str) -> bool:
        """Send a message on the socket. If none is initialized, return False."""
        if self._status_server is None:
            return False
        data = msg.encode("UTF-8")
        length_message = (
            str(len(msg)).encode("UTF-8").rjust(STATUS_SERVER_MESSAGE_LENGTH_BYTES, b"0")
        )
        self._status_server.sendall(length_message)
        self._status_server.sendall(data)
        return True

    @staticmethod
    def _message_template_json(message_type: MessageType, data: Any) -> str:
        return json.dumps({"message_type": message_type, "data": data})

    def update_status(self, status: RunStatus) -> bool:
        """Send a status update to the server. Return if data was sent successfully."""
        msg = _StatusServer._message_template_json(MessageType.STATUS_UPDATE, {"status": status})
        return self._send_message(msg)

    def update_step(self, step: str) -> bool:
        """Send a step update to the server. Return if data was sent successfully."""
        msg = _StatusServer._message_template_json(
            MessageType.CURRENT_STEP_UPDATE, {"current_step": step}
        )
        return self._send_message(msg)

    def close_status_socket(self) -> None:
        """Close the existing connection to the status server if there is one."""
        if self._status_server is not None:
            self._status_server.close()
            self._status_server = None


status_server = _StatusServer()
