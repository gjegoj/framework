"""Future Lightning callbacks: EMA, freezing, scheduling and reporting.

Callbacks use Lightning's existing hooks. No second callback/event framework is introduced.
Weight-changing callbacks run before checkpointing; sample reporting delegates to integrations.
"""
