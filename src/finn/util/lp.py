"""Shared helpers for building `python-mip` linear programs.

FINN+ solves more than one assignment problem with an LP (multi-FPGA device
partitioning, SLR floorplanning). They do not share a problem formulation, but they
do share the mechanics of getting a usable `mip.Model` out of the installed solvers,
including two environment-specific workarounds that are easy to get wrong when
duplicated. Those live here so there is exactly one copy.
"""

import locale
import mip
import mip.exceptions
import os
import random
import time
from mip import Model
from typing import Any, cast

from finn.builder.build_dataflow_config import MIPSolver
from finn.util.exception import FINNUserError
from finn.util.logging import log

# Gurobi hands out one floating license token per environment and asks its token server for it
# every time an environment is created. In a highly parallel setting (the CI test suite runs
# dozens of pytest workers that all create partitioners, and every floorplanned build solves an
# SLR assignment) this fails now and then because the token server is momentarily out of tokens
# or does not answer in time. Gurobi's own recommendation for this case is to retry after a
# short delay, see `create_model_with_retries`.
MIP_SOLVER_MAX_ATTEMPTS_ENV = "FINN_MIP_SOLVER_MAX_ATTEMPTS"  # overrides the default below
MIP_SOLVER_MAX_ATTEMPTS = 6
MIP_SOLVER_RETRY_DELAY = 5.0  # seconds before the 2nd attempt, doubled after each further one
MIP_SOLVER_RETRY_DELAY_MAX = 60.0  # seconds


def create_mip_model(
    solver: MIPSolver | None, name: str, error_type: type[FINNUserError] = FINNUserError
) -> mip.Model:
    """Create a `mip.Model`, falling back to CBC when the default solver is unusable.

    Args:
        solver: The solver to use. `None` lets `mip` pick its default and falls back to
            CBC if that fails; an explicit solver is treated as a user choice, retried on
            transient failures (`create_model_with_retries`) and otherwise raises instead
            of falling back silently.
        name: Name given to the model, used in solver log output and dumps.
        error_type: The `FINNUserError` subclass raised when an explicit solver cannot be
            instantiated (the multi-FPGA partitioner raises its own type).

    Returns:
        A `mip.Model` ready to have variables and constraints added to it.

    Raises:
        FINNUserError: If an explicitly requested solver cannot be instantiated.
    """
    # A failed Gurobi model instantiation can switch the default locale/encoding away
    # from UTF-8, which then breaks unrelated file IO later in the build. Snapshot the
    # locale and restore it afterwards.
    current_locale = locale.getlocale(locale.LC_CTYPE)
    try:
        if solver is not None:
            return create_model_with_retries(solver.value, name, error_type)
        try:
            return Model()
        except OSError:
            log.warning(
                "Creation of mip.Model failed. This might be known bug "
                "(LD_LIBRARY_PATH only modified at runtime to point to "
                "libgurobi instead of before). Falling back to CBC."
            )  # See finn-plus issue #67
            return Model(name=name, solver_name=mip.CBC)
        except mip.exceptions.InterfacingError as e:
            log.warning(
                f"Could not create a default-initialized mip.Model. "
                f"The error encountered was: {e}. Trying to fallback to a CBC based model."
            )
            return Model(name=name, solver_name=mip.CBC)
    finally:
        locale.setlocale(locale.LC_CTYPE, current_locale)


def create_model_with_retries(
    solver_name: str, name: str, error_type: type[FINNUserError] = FINNUserError
) -> mip.Model:
    """Create the ``mip.Model`` for the given solver, retrying transient failures.

    Creating the model fails transiently for Gurobi when its floating license token server
    is out of tokens or unreachable (see the module comment). Such failures are retried with
    an exponential backoff and a bit of jitter so that many parallel processes do not all hit
    the server again at the same moment. Any other failure, and a transient failure that
    persists for all attempts, raises ``error_type`` with Gurobi's own reason when it can be
    determined.
    """
    try:
        max_attempts = max(1, int(os.environ.get(MIP_SOLVER_MAX_ATTEMPTS_ENV, "")))
    except ValueError:
        max_attempts = MIP_SOLVER_MAX_ATTEMPTS
    delay = MIP_SOLVER_RETRY_DELAY
    for attempt in range(1, max_attempts + 1):
        try:
            return Model(name=name, solver_name=solver_name)
        # A retry loop: the try/except has to be inside it.
        except mip.exceptions.InterfacingError as e:  # noqa: PERF203
            # Only a failed Gurobi environment creation can disappear on retry (no token or
            # no answer from the token server right now). A missing solver library or a
            # failed model creation is a persistent setup problem.
            transient = "environment could not be loaded" in str(e)
            if transient and attempt < max_attempts:
                log.warning(
                    f"Creating the {solver_name} solver environment failed (attempt "
                    f"{attempt}/{max_attempts}): {e} Retrying in {delay:.1f}s."
                )
                time.sleep(delay)
                # Exponential backoff with +-25% jitter, capped.
                delay = min(delay * 2, MIP_SOLVER_RETRY_DELAY_MAX) * random.uniform(0.75, 1.25)
                continue
            message = f"Cannot create mip solver of type {solver_name}"
            if transient:
                # Now that we give up anyway, ask Gurobi once for the actual reason.
                reason = gurobi_environment_error() if solver_name == mip.GUROBI else None
                details = f" ({reason})" if reason else ""
                message += (
                    f" after {attempt} attempt(s). Original error: {e}{details} If this "
                    f"is a floating license, the token server may be out of tokens or "
                    f"unreachable; the number of attempts can be set with "
                    f"{MIP_SOLVER_MAX_ATTEMPTS_ENV}."
                )
            else:
                message += f". Original error: {e}"
            raise error_type(message) from e
    raise AssertionError("unreachable")  # pragma: no cover


# The extra Gurobi functions used by gurobi_environment_error are declared in python-mip's
# cffi binding once per process.
_gurobi_diagnostics_declared = False


def gurobi_environment_error() -> str | None:
    """Ask Gurobi why creating an environment currently fails.

    python-mip reports every failed environment creation as "check your license", which
    hides the real reason (out of tokens, token server unreachable, expired license, ...).
    This creates an empty environment through the same cffi binding python-mip uses, starts
    it, and returns Gurobi's own error message with its error code. Starting the environment
    requests a token like any other attempt, which is why this is only done once, when
    giving up. Returns ``None`` if the environment starts fine (the failure was transient and
    is already gone) or if the diagnostics cannot be collected at all.
    """
    global _gurobi_diagnostics_declared
    try:
        from mip import gurobi as mip_gurobi

        if not mip_gurobi.has_gurobi:
            return None
        ffi = mip_gurobi.ffi
        # The functions below are declared at runtime via ffi.cdef, so the cffi stubs
        # cannot know them.
        lib: Any = mip_gurobi.grblib
        if not _gurobi_diagnostics_declared:
            ffi.cdef(
                """
                int GRBemptyenv(GRBenv **envP);
                int GRBstartenv(GRBenv *env);
                const char *GRBgeterrormsg(GRBenv *env);
                """
            )
            _gurobi_diagnostics_declared = True
        env_ptr = ffi.new("GRBenv **")
        status = lib.GRBemptyenv(env_ptr)
        if status != 0:
            return f"Gurobi error {status} while creating an empty environment"
        env = env_ptr[0]
        try:
            # A failed GRBstartenv leaves the environment as it was, so the error message
            # can be read from it and it must still be freed.
            status = lib.GRBstartenv(env)
            if status == 0:
                return None
            # ffi.string of a "const char *" is always bytes.
            message = cast("bytes", ffi.string(lib.GRBgeterrormsg(env))).decode(
                "utf-8", errors="replace"
            )
            return f"Gurobi error {status}: {message.strip()}"
        finally:
            lib.GRBfreeenv(env)
    except Exception as e:  # diagnostics must never mask the original error
        log.debug(f"Could not collect Gurobi diagnostics: {e}")
        return None
