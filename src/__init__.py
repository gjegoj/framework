"""The framework's packages; the root imports none of them.

Import a contract from the package that owns it — ``src.models``, ``src.tasks``, ``src.training``.
Importing a package is what registers the names its declarations may write, so the root deliberately
initializes nothing: a tool that only reads a schema pays for no model hub.
"""
