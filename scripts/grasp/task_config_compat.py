"""Workaround for resolving the cloth task config on the shared GPU server.

Isaac Lab converts ``slice`` values to strings before handing the config to
OmegaConf, but that conversion does not go inside tuples. The cloth task's
``reset_deformable`` event stores ``support_cfg`` as a tuple of
``SceneEntityCfg`` whose default ``joint_ids`` is ``slice(None)``, so
``resolve_task_config`` fails with ``UnsupportedValueType: Value 'slice'``.

This module only changes the copy that OmegaConf receives, and only while the
config is being resolved. The returned env config object is not modified here.
"""
from __future__ import annotations

from contextlib import contextmanager


def _omegaconf_safe(value):
    """Return a copy with tuples as lists and slices as Isaac Lab slice strings."""
    if isinstance(value, slice):
        return f"slice({value.start},{value.stop},{value.step})"
    if isinstance(value, dict):
        return {key: _omegaconf_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_omegaconf_safe(item) for item in value]
    return value


@contextmanager
def omegaconf_slice_in_tuple_fix():
    """Temporarily sanitize the input of ``OmegaConf.create``."""
    from omegaconf import OmegaConf

    original = OmegaConf.create

    def create(*args, **kwargs):
        if args:
            args = (_omegaconf_safe(args[0]), *args[1:])
        elif "obj" in kwargs:
            kwargs["obj"] = _omegaconf_safe(kwargs["obj"])
        return original(*args, **kwargs)

    OmegaConf.create = create
    try:
        yield
    finally:
        OmegaConf.create = original


def resolve_cloth_task_config(task: str, agent: str = ""):
    """``resolve_task_config`` with the tuple-slice workaround applied."""
    from isaaclab_tasks.utils import resolve_task_config

    with omegaconf_slice_in_tuple_fix():
        return resolve_task_config(task, agent)
