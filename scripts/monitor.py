"""Script to monitor progess of FINN+ runs. Status data is collected via a socket."""
# ruff: noqa: E701
from __future__ import annotations

import argparse
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import RLock, Thread
from typing import Any

from finn.util.logging import STATUS_SERVER_MESSAGE_LENGTH_BYTES, RunStatus

# TODO: Logging with logging instead of print
# TODO: Curses based interface
# TODO: Support network sockets


class ConnectionLogBuffer:
    """Log infos for all connections. Thread safe. Connection -1 is reserved."""

    def __init__(self) -> None:
        """Create a new ConnectionLogger."""
        self._buffer: dict[int, str] = {-1: ""}
        self._lock: RLock = RLock()

    def add_connection(self, index: int) -> None:
        """Add a new connection to log. If it already exists, do nothing."""
        with self._lock:
            if index not in self._buffer and index != -1:
                self._buffer[index] = ""

    def log_status(self, msg: str) -> None:
        """Log to the special status logger. Thread safe."""
        with self._lock:
            self._buffer[-1] += msg

    def log(self, index: int, msg: str) -> None:
        """Log the message. If the index is wrong or reserved, a status message is written."""
        with self._lock:
            if index == -1 or index not in self._buffer:
                self.log_status(
                    f"Tried logging to index {index} which is "
                    f"either reserved (-1) or not added!\n"
                )
            else:
                self._buffer[index] += msg

    def pop(self, index: int) -> str:
        """Read the logs of the given connection index. It is empty afterwards."""
        with self._lock:
            msg = ""
            if index == -1 or index not in self._buffer:
                self.log_status(
                    f"Tried reading from connection index {index} which "
                    f"is either reserved (-1) or not added!"
                )
            else:
                msg = self._buffer[index]
                self._buffer[index] = ""
        return msg

    def pop_all(self) -> dict[int, str]:
        """Return the buffered outputs of all connections and remove them from the buffer."""
        with self._lock:
            data = deepcopy(self._buffer)
            for key in self._buffer.keys():
                self._buffer[key] = ""
            return data


class ConnectionData:
    """Keep data for all connections. Threadsafe."""

    def __init__(self, logger: ConnectionLogBuffer) -> None:
        """Create a new data object."""
        self._data: dict[int, dict[str, Any]] = {}
        self._lock: RLock = RLock()
        self._logger: ConnectionLogBuffer = logger

    def add_connection(self, index: int) -> None:
        """Add a new connection. If it already exists, do nothing."""
        if index == -1:
            self._logger.log_status("Cannot add connection at index -1. Reserved.")
            return
        with self._lock:
            if index in self._data:
                return
            self._data[index] = {"status": RunStatus.UNKNOWN, "current_step": None}

    def assert_field(self, index: int, field: str, data: dict | None = None) -> bool:
        """Check if the given field is in the data of connection with given index.

        If the field data is None, the internal data is used. Otherwise the given data.
        """
        with self._lock:
            if index not in self._data:
                self._logger.log_status(
                    f"Tried assert field {field} in index {index}. "
                    f"But this index does not exist!"
                )
                return False
            if data is None:
                return field in self._data[index].keys()
            return field in data.keys()

    def update_field(self, index: int, field: str | list[str], data: Any) -> None:
        """Update the given field."""
        with self._lock:
            if index not in self._data:
                self._logger.log_status(
                    f"Cannot update field {field} in " f"non existent index {index}!"
                )
                return
            if type(field) is str:
                if field in self._data[index]:
                    self._data[index][field] = data
                else:
                    self._logger.log_status(
                        f"Tried updating non existent " f"field {field} at index {index}."
                    )
            else:
                current = self._data[index]
                while len(field) > 1:
                    try:
                        current = current[field[0]]
                        field = field[1:]
                    except Exception:  # noqa
                        self._logger.log_status(f"Cannot access field {field} on index {index}!")
                        return
                current[field[0]] = data

    def get_field(self, index: int, field: str | list[str]) -> Any | None:
        """Get the data at the field of the given connection."""
        with self._lock:
            if index not in self._data:
                self._logger.log_status(
                    f"Cannot update field {field} in " f"non existent index {index}!"
                )
                return None

            # String case
            if type(field) is str:
                if field in self._data[index]:
                    return self._data[index][field]
                self._logger.log_status(
                    f"Tried updating non existent " f"field {field} at index {index}."
                )
                return None

            # Nested arg case
            current = self._data[index]
            while len(field) > 0:
                try:
                    current = current[field[0]]
                    field = field[1:]
                except Exception:  # noqa
                    self._logger.log_status(f"Cannot access field {field} on index {index}!")
                    return None
            return current

    def status(self, index: int) -> str:
        """Get a string status update for the given connection."""
        with self._lock:
            if index not in self._data:
                return f"Cannot get status for non-existent connection {index}!"
            s = f"Status of connection {index}:\n"
            for field, data in self._data[index].items():
                s += f"\t{field}: {data}\n"
            return s

    def get_indices(self) -> list[int]:
        """Get a list of all indices."""
        with self._lock:
            return list(self._data.keys())

    def close_connection(self, index: int) -> None:
        """Remove data for a closed connection. If the index doesn't exist do nothing."""
        with self._lock:
            if index in self._data:
                del self._data[index]


LOG = ConnectionLogBuffer()
DATA = ConnectionData(LOG)


def handle_connection(index: int, client: socket.socket) -> None:
    """Handle a connecting FINN+ client."""
    global LOG, DATA

    # Initialize DATA for the given connection index
    DATA.add_connection(index)
    check_field = lambda data, field: DATA.assert_field(index, field, data)  # noqa
    update_data = lambda field, dat: DATA.update_field(index, field, dat)  # noqa
    LOG.add_connection(index)
    log = lambda msg: LOG.log(index, msg + "\n")  # noqa

    # Status for the user
    log(f"Client with connection ID {index} connected and initialized!")

    # Check for incoming data
    while True:
        # Receive a fixed width byte header. For now only for the following length
        bytes_data = client.recv(STATUS_SERVER_MESSAGE_LENGTH_BYTES)

        # Close connection if no data is there
        if not bytes_data:
            DATA.close_connection(index)
            log(f"Connection on index {index} closed.")
            return

        # Handle the actual message
        expect_bytes = int(bytes_data.decode("UTF-8"))
        message = client.recv(expect_bytes).decode("UTF-8")
        if not message:
            DATA.close_connection(index)
            log(f"Connection on index {index} closed.")
            return
        log(f"[{index}]: {message}")
        parsed = json.loads(message)
        if not check_field(None, "message_type"):
            continue
        if not check_field(None, "data"):
            continue
        msg_type = parsed["message_type"]
        message_data = parsed["data"]

        # Match on the specific message passed
        match msg_type:
            case "current_step_update":
                if not check_field(message_data, "current_step"):
                    continue
                update_data("current_step", message_data["current_step"])
            case "status_update":
                if not check_field(message_data, "status"):
                    continue
                update_data("status", RunStatus(message_data["status"]))
            case _:
                log(f"Unknown message type: {msg_type}")


def user_interaction(mode: str, poll_pause: float) -> None:
    """Handle user interaction. Either via a text-based interface or curses."""
    global DATA, LOG
    if mode == "tui":
        raise NotImplementedError("TUI mode not implemented yet.")
    elif mode == "live":
        last_data = {}
        while True:
            time.sleep(poll_pause)
            bufferdata = LOG.pop_all()
            if bufferdata != last_data:
                for index, data in bufferdata.items():
                    if data != "":
                        print(f"(buffered) {index}: {data}")
                last_data = bufferdata
    elif mode == "text":
        while True:
            cmd = input("> ")
            cmd_initial = cmd.split(" ")[0]
            match cmd_initial:
                case "status":
                    for index in DATA.get_indices():
                        print(DATA.status(index))
                case "help":
                    print("Commands: status, help")
                case _:
                    print(
                        f'Unknown command: {cmd_initial}. Enter "help" to get '
                        f"an overivew of all available commands!"
                    )

            # In curses version this can be done in parallel to accepting input
            bufferdata = LOG.pop_all()
            for index, data in bufferdata.items():
                print(f"(buffered) {index}: {data}")
    else:
        raise NotImplementedError(f"Mode {mode} not implemented.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("socket", help="Socket / address that FINN+ clients communicate with")
    parser.add_argument(
        "--max-connections",
        "-c",
        default=1000,
        type=int,
        help="Number of connections that the server can handle concurrently",
    )
    parser.add_argument("--mode", "-m", help="Mode. tui or text.", default="text")
    parser.add_argument(
        "--poll-pause",
        "-p",
        default=0.5,
        type=float,
        help="Number of seconds (float) to wait between polling the "
        "status of the local data fields.",
    )
    args = parser.parse_args()

    # Remove socket if it already exists
    socket_target = Path(args.socket)
    if socket_target.exists():
        socket_target.unlink()

    print(f"Using socket {socket_target}")
    index = 0
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        # Bind socket
        server.bind(str(socket_target))
        print(f"Socket bound. Listening. Max connections: {args.max_connections}")
        server.listen(args.max_connections)

        # Start thread for user interaction
        user_thread = Thread(
            target=user_interaction, kwargs={"mode": args.mode, "poll_pause": args.poll_pause}
        )
        user_thread.start()

        # Accept connections and dispatch handlers in main thread
        with ThreadPoolExecutor(max_workers=args.max_connections) as pool:
            while True:
                conn, _ = server.accept()
                pool.submit(handle_connection, index=index, client=conn)
                index += 1
        user_thread.join()
