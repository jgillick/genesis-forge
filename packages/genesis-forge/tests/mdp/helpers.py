"""Shared by the MDP function tests."""


def built(fn, env, **context):
    """Bind an MDP function to the environment and build it, as its manager would."""
    fn.context(env, **context)
    fn.safe_build()
    return fn
