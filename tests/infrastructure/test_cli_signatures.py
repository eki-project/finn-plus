"""Static consistency check of the ``finn`` command line interface."""

import pytest

import click
import inspect

import finn.interface.run_finn as run_finn


def _commands() -> list[click.Command]:
    """Return every click command (not group) defined in the CLI module."""
    return [
        obj
        for obj in vars(run_finn).values()
        if isinstance(obj, click.Command) and not isinstance(obj, click.Group)
    ]


@pytest.mark.infrastructure
@pytest.mark.parametrize("command", _commands(), ids=lambda c: c.name)
def test_callback_accepts_all_declared_parameters(command: click.Command) -> None:
    """Click passes every declared option/argument to the callback by name, so a callback
    that lacks one of them crashes with a TypeError as soon as the command is invoked."""
    assert command.callback is not None
    accepted = inspect.signature(command.callback).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return
    declared = {param.name for param in command.params}
    assert declared <= set(accepted), (
        f"'finn {command.name}' declares parameters its function does not accept: "
        f"{sorted(declared - set(accepted))}"
    )
