#!/usr/bin/env python3
"""Backward-compatible CLI for diagnostic-only expert screening."""

from agent_system.environments.env_package.awm.screening.expert import *  # noqa: F401,F403

if __name__ == "__main__":
    main()  # noqa: F405
