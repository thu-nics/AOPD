#!/usr/bin/env python3
"""CLI wrapper for AWM expert qualification."""

from agent_system.environments.env_package.awm.qualification import *  # noqa: F401,F403
from agent_system.environments.env_package.awm.qualification import _load_jsonl  # noqa: F401

if __name__ == "__main__":
    main()
