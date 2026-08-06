#!/usr/bin/env python3
"""CLI wrapper for standalone AWM-native evaluation."""

from agent_system.environments.env_package.awm.evaluation.native import *  # noqa: F401,F403
from agent_system.environments.env_package.awm.evaluation.native import (  # noqa: F401
    _fit_context,
    _model_artifact_identity,
)

if __name__ == "__main__":
    main()
