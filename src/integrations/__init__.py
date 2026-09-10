"""Optional bridges. Only these modules may combine framework and visualization APIs.

The visualization bridge converts requested detached samples to the display library's
own types. Importing framework must not load that library or allocate previews.
"""
