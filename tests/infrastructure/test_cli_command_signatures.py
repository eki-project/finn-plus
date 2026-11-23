"""Tests that every `finn` command accepts the parameters its click options produce."""

import pytest

import click
import inspect

from finn.interface import run_finn

COMMANDS = {
    name: obj
    for name, obj in vars(run_finn).items()
    if isinstance(obj, click.Command) and obj.callback is not None
}


@pytest.mark.infrastructure
@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_cli_callback_accepts_its_click_parameters(name: str) -> None:
    """click passes every option and argument of a command to its callback by name, so a
    callback parameter that is named differently makes the command fail on every call
    (as `finn run` did with `num_workers` instead of `num_default_workers`).
    """
    command = COMMANDS[name]
    click_params = {param.name for param in command.params}
    callback_params = set(inspect.signature(command.callback).parameters)
    assert click_params == callback_params
