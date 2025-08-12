#!/usr/bin/env python3
"""
Fix for Python 3.11 compatibility with vulkan package
Patches collections.Iterable to collections.abc.Iterable
"""

import collections
import collections.abc
import sys

# Monkey patch for Python 3.11 compatibility
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
    collections.Iterator = collections.abc.Iterator
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
    collections.Sequence = collections.abc.Sequence
    collections.MutableSequence = collections.abc.MutableSequence
    collections.Set = collections.abc.Set
    collections.MutableSet = collections.abc.MutableSet
    collections.Callable = collections.abc.Callable
    print("✅ Applied Python 3.11 compatibility patches for vulkan")

# Now we can import vulkan without errors
import vulkan