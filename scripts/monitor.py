"""Script to monitor progess of FINN+ runs. Status data is collected via a socket."""
# ruff: noqa: E701
from __future__ import annotations

import argparse
import datetime
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from threading import RLock, Thread
from typing import Any

from finn.util.logging import STATUS_SERVER_MESSAGE_LENGTH_BYTES, MessageType, RunStatus

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
        self._finished_runs: dict[int, dict[str, Any]] = {}

    def add_connection(self, index: int) -> None:
        """Add a new connection. If it already exists, do nothing."""
        if index == -1:
            self._logger.log_status("Cannot add connection at index -1. Reserved.")
            return
        with self._lock:
            if index in self._data:
                return
            self._data[index] = {
                "status": RunStatus.UNKNOWN,
                "current_step": None,
                "config": None,
                "model": None,
                "last_update": datetime.datetime.now(),
                "hostname": None,
            }

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
                    f"Cannot update field {field} in non existent index {index}!"
                )
                return
            if type(field) is str:
                self._data[index]["last_update"] = datetime.datetime.now()
                self._data[index][field] = data
            else:
                current = self._data[index]
                while len(field) > 1:
                    try:
                        current = current[field[0]]
                        field = field[1:]
                    except Exception:  # noqa
                        self._logger.log_status(f"Cannot access field {field} on index {index}!")
                        return
                self._data[index]["last_update"] = datetime.datetime.now()
                current[field[0]] = data

    def get_field(self, index: int, field: str | list[str]) -> Any | None:
        """Get the data at the field of the given connection."""
        with self._lock:
            if index not in self._data:
                self._logger.log_status(
                    f"Cannot update field {field} in non existent index {index}!"
                )
                return None

            # String case
            if type(field) is str:
                if field in self._data[index]:
                    return self._data[index][field]
                self._logger.log_status(
                    f"Tried updating non existent field {field} at index {index}."
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
        """Remove data for a closed connection. If the index doesn't exist do nothing.

        The last updated data is added to a table for inspection.
        """
        with self._lock:
            if index in self._data:
                self._finished_runs[index] = deepcopy(self._data[index])
                del self._data[index]

    def _make_rich_table(self, data_dict: dict, is_live: bool) -> Table:
        """Create a rich table representation of the current status."""
        status_color = {
            RunStatus.UNKNOWN: "orange3",
            RunStatus.SUCCESS: "green3",
            RunStatus.FAIL: "red3",
            RunStatus.RUNNING: "grey85",
            RunStatus.KEYBOARD_INTERRUPT: "magenta3",
        }
        status_to_color = lambda status: (  # noqa
            f"[{status_color[status]}]{status}[/{status_color[status]}]"
            if status is not None
            else "-"
        )
        table = Table()
        status_header = "Status" if is_live else "Last known status"
        table.add_column("ID", header_style="italic bold grey82", style="bold grey82")
        table.add_column("Host", header_style="italic bold grey82", style="bold grey82")
        table.add_column("Config", header_style="italic dark_cyan", style="dark_cyan")
        table.add_column("Model", header_style="italic slate_blue3", style="slate_blue3")
        table.add_column(status_header, header_style="italic")
        table.add_column("Step", header_style="italic yellow3", style="yellow3")
        table.add_column(
            "Last updated", header_style="italic medium_purple3", style="medium_purple3"
        )
        truncate = lambda txt, width: "..." + str(txt)[-width:] if txt is not None else None  # noqa
        none_to_str = lambda msg: msg if msg is not None else "-"  # noqa
        with self._lock:
            for index, data in data_dict.items():
                table.add_row(
                    str(index),
                    none_to_str(data["hostname"]),
                    none_to_str(truncate(data["config"], 45)),
                    none_to_str(truncate(data["model"], 45)),
                    status_to_color(data["status"]),
                    none_to_str(data["current_step"]),
                    data["last_update"].strftime("%d.%m.%Y  %H:%M"),
                )
        table.box = box.MINIMAL
        return table

    def make_live_table(self) -> Table:
        """Return a rich table of currently running flows."""
        with self._lock:
            return self._make_rich_table(self._data, is_live=True)

    def make_finished_run_table(self) -> Table:
        """Return a rich table of past runs."""
        with self._lock:
            return self._make_rich_table(self._finished_runs, is_live=False)

    def open_connections(self) -> int:
        """Return number of open connections."""
        with self._lock:
            return len(self._data)


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
        msg_type = parsed["message_type"]
        message_data = parsed["data"]

        # Match on the specific message passed
        match msg_type:
            case MessageType.CURRENT_STEP_UPDATE:
                if "current_step" not in message_data:
                    continue
                step = message_data["current_step"]
                update_data("current_step", step)
            case MessageType.STATUS_UPDATE:
                if "status" not in message_data:
                    continue
                status = message_data["status"]
                update_data("status", RunStatus(message_data["status"]))
            case MessageType.INTRODUCTION:
                if (
                    "model" not in message_data
                    or "status" not in message_data
                    or "hostname" not in message_data
                ):
                    continue
                model = Path(message_data["model"])
                status = message_data["status"]
                hostname = message_data["hostname"]
                update_data("model", model)
                update_data("status", status)
                update_data("hostname", hostname)
                if "config" in message_data:
                    config = Path(message_data["config"])
                    update_data("config", config)
            case _:
                log(f"Unknown message type: {msg_type}")


def user_interaction(mode: str, poll_pause: float) -> None:
    """Handle user interaction. Either via a text-based interface or curses."""
    global DATA, LOG
    if mode == "tui":
        raise NotImplementedError("TUI mode not implemented yet.")
    elif mode == "log":
        last_data = {}
        while True:
            time.sleep(poll_pause)
            bufferdata = LOG.pop_all()
            if bufferdata != last_data:
                for index, data in bufferdata.items():
                    if data != "":
                        print(f"(buffered) {index}: {data}")
                last_data = bufferdata
    elif mode == "table":
        console = Console()

        def create_layout() -> Layout:
            live_table = DATA.make_live_table()
            live_table.title = "Live FINN+ Runs"
            live_table.caption = (
                f"Time between polls: {poll_pause}s. Listening on "
                f"{socket_target}. Open connections: {DATA.open_connections()}"
            )
            live_table.caption_style = "italic grey50"
            finished_table = DATA.make_finished_run_table()
            finished_table.title = "Finished FINN+ Runs"
            layout = Layout()
            layout.split_column(Align.center(live_table), Align.center(finished_table))
            return layout

        with Live(create_layout(), console=console, refresh_per_second=20, screen=True) as live:
            while True:
                time.sleep(poll_pause)
                live.update(create_layout())
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
    parser.add_argument("--mode", "-m", help="log / table / tui", default="text")
    parser.add_argument(
        "--poll-pause",
        "-p",
        default=0.1,
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
