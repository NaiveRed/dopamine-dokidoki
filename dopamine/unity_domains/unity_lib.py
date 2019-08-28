# coding=utf-8
"""Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which
is in charge of:
  . Emitting a terminal signal when losing a life (optional).
  . Frame skipping and color pooling.
  . Resizing the image before it is provided to the agent.
"""
import gin

from dopamine.unity_domains.unity_wrappers import wrap_unity_env


@gin.configurable
def create_unity_environment(game_path=None):
    """Wrap the unity environment
  """
    assert game_path is not None and game_path[-4:] == ".exe", "Game path error."
    env = wrap_unity_env(game_path, port=9538, use_visual=False)
    return env
