#!/usr/bin/env python3
"""CLI wrapper for pinned AWM dataset preparation."""

from agent_system.environments.env_package.awm.data.prepare import *  # noqa: F401,F403
from agent_system.environments.env_package.awm.data.prepare import (  # noqa: F401
    _select_splits,
    _validate_and_expand,
    _validate_source_hashes,
)

if __name__ == "__main__":
    main()
