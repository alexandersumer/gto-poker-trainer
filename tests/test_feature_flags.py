from __future__ import annotations

import os

from gtotrainer.core import feature_flags


def test_env_and_override_stack() -> None:
    env_var = "GTOTRAINER_FEATURES"
    original = os.environ.get(env_var)
    try:
        if env_var in os.environ:
            del os.environ[env_var]

        assert feature_flags.is_enabled("solver.high_precision_cfr") is False

        feature_flags.set_env_flags(["solver.high_precision_cfr"])
        assert feature_flags.is_enabled("solver.high_precision_cfr") is True

        with feature_flags.override(disable={"solver.high_precision_cfr"}):
            assert feature_flags.is_enabled("solver.high_precision_cfr") is False
            with feature_flags.override(enable={"rival.texture_v2"}):
                assert feature_flags.is_enabled("rival.texture_v2") is True
                assert feature_flags.is_enabled("solver.high_precision_cfr") is False

        assert feature_flags.is_enabled("solver.high_precision_cfr") is True

    finally:
        if original is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = original
