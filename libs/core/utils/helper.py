#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

from typing import Optional
from importlib import import_module


def get_class_by_name(class_name: str,
                      default_module_name: Optional[str] = None) -> type:
    """Load class by its name

    Parameters
    ----------
    class_name : `str`
    default_module_name : `str`, optional
        When provided and `class_name` does not contain the absolute path

    Returns
    -------
    Klass : `type`
        Class.

    Example
    -------
    >>> ClopiNet = get_class_by_name(
    ...     pyannoteNet')
    >>> ClopiNet = get_class_by_name(
    ...     'ClopiNet', default_module_npyannote.models')
    """
    tokens = class_name.split('.')

    if len(tokens) == 1:
        if default_module_name is None:
            msg = (
                f'Could not infer module name from class name "{class_name}".'
                f'Please provide default module name.')
            raise ValueError(msg)
        module_name = default_module_name
    else:
        module_name = '.'.join(tokens[:-1])
        class_name = tokens[-1]

    return getattr(import_module(module_name), class_name)
