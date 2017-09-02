#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal
======

Defines the class implementing support for continuous signal data
representation:

-   :class:`Signal`
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import Iterator, Mapping, OrderedDict, Sequence
from operator import add, mul, pow, sub, iadd, imul, ipow, isub

# Python 3 compatibility.
try:
    from operator import div, idiv
except ImportError:
    from operator import truediv, itruediv

    div = truediv
    idiv = itruediv

from colour.algebra import Extrapolator, LinearInterpolator
from colour.continuous import AbstractContinuousFunction
from colour.utilities import (as_numeric, closest, fill_nan,
                              is_pandas_installed, ndarray_write, tsplit,
                              tstack, warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Signal']


class Signal(AbstractContinuousFunction):
    """
    Defines the base class for continuous signal data representation.

    The implementation builds on an interpolating function encapsulated inside
    an extrapolating function. The function independent domain, stored as
    discrete values in the :attr:`Signal.domain` attribute corresponds with the
    function dependent and already known range stored in the
    :attr:`Signal.range`.

    Parameters
    ----------
    data : Series or Signal or array_like or dict_like, optional
        Data to be stored in the :class:`Signal` class instance.
    domain : array_like, optional
        Values to initialise the :attr:`Signal.domain` attribute with.
        If both `data` and `domain` arguments are defined, the latter with be
        used to initialise the :attr:`Signal.domain` attribute.

    Other Parameters
    ----------------
    name : unicode, optional
        Continuous data representation name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating extrapolating function.

    Attributes
    ----------
    domain
    range
    interpolator
    interpolator_args
    extrapolator
    extrapolator_args
    function

    Methods
    -------
    __str__
    __repr__
    __getitem__
    __setitem__
    __contains__
    __eq__
    __neq__
    arithmetical_operation
    signal_unpack_data
    fill_nan
    uncertainty

    Examples
    --------
    """

    def __init__(self, data=None, domain=None, **kwargs):
        super(Signal, self).__init__(kwargs.get('name'))

        self._domain = None
        self._range = None
        self._interpolator = LinearInterpolator
        self._interpolator_args = {}
        self._extrapolator = Extrapolator
        self._extrapolator_args = {
            'method': 'Constant',
            'left': np.nan,
            'right': np.nan
        }

        self.domain, self.range = self.signal_unpack_data(data, domain)

        self.interpolator = kwargs.get('interpolator')
        self.interpolator_args = kwargs.get('interpolator_args')
        self.extrapolator = kwargs.get('extrapolator')
        self.extrapolator_args = kwargs.get('extrapolator_args')

        self._create_function()

    @property
    def domain(self):
        """
        Getter for **self.domain** property.

        Returns
        -------
        ndarray
            BLABLABLALBAL
        """

        return self._domain

    @domain.setter
    def domain(self, value):
        if value is not None:
            if not np.all(np.isfinite(value)):
                warning('"domain" variable is not finite, '
                        'unpredictable results may occur!\n{0}'.format(value))

            # TODO: `self.domain` is a copy of `value` to avoid side effects,
            # Is it a smart way to avoid them?
            value = np.copy(np.asarray(value))

            if self._range is not None:
                assert value.size == self._range.size, (
                    '"domain" and "range" variables must have same size!')

            value.setflags(write=False)
            self._domain = value
            self._create_function()

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, value):
        if value is not None:
            if not np.all(np.isfinite(value)):
                warning('"range" variable is not finite, '
                        'unpredictable results may occur!\n{0}'.format(value))

            # TODO: `self.range` is a copy of `value` to avoid side effects,
            # Is it a smart way to avoid them?
            value = np.copy(np.asarray(value))

            if self._domain is not None:
                assert value.size == self._domain.size, (
                    '"domain" and "range" variables must have same size!')

            value.setflags(write=False)
            self._range = value
            self._create_function()

    @property
    def interpolator(self):
        return self._interpolator

    @interpolator.setter
    def interpolator(self, value):
        if value is not None:
            # TODO: Check for interpolator capabilities.
            self._interpolator = value
            self._create_function()

    @property
    def interpolator_args(self):
        return self._interpolator_args

    @interpolator_args.setter
    def interpolator_args(self, value):
        if value is not None:
            assert type(value) in (dict, OrderedDict), ((
                '"{0}" attribute: "{1}" type is not '
                '"dict" or "OrderedDict"!').format('interpolator_args', value))

            self._interpolator_args = value
            self._create_function()

    @property
    def extrapolator(self):
        return self._extrapolator

    @extrapolator.setter
    def extrapolator(self, value):
        if value is not None:
            # TODO: Check for extrapolator capabilities.
            self._extrapolator = value
            self._create_function()

    @property
    def extrapolator_args(self):
        return self._extrapolator_args

    @extrapolator_args.setter
    def extrapolator_args(self, value):
        if value is not None:
            assert type(value) in (dict, OrderedDict), ((
                '"{0}" attribute: "{1}" type is not '
                '"dict" or "OrderedDict"!').format('extrapolator_args', value))

            self._extrapolator_args = value
            self._create_function()

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value):
        raise AttributeError(
            '"{0}" attribute is read only!'.format('function'))

    def __str__(self):
        try:
            return str(tstack((self.domain, self.range)))
        except TypeError:
            return super(Signal, self).__str__()

    def __repr__(self):
        try:
            representation = repr(tstack((self.domain, self.range)))
            representation = representation.replace('array',
                                                    self.__class__.__name__)
            representation = representation.replace('       [', '{0}['.format(
                ' ' * (len(self.__class__.__name__) + 2)))
            representation = ('{0},\n'
                              '{1}interpolator={2},\n'
                              '{1}interpolator_args={3},\n'
                              '{1}extrapolator={4},\n'
                              '{1}extrapolator_args={5})').format(
                                  representation[:-1],
                                  ' ' * (len(self.__class__.__name__) + 1),
                                  self.interpolator.__name__,
                                  repr(self.interpolator_args),
                                  self.extrapolator.__name__,
                                  repr(self.extrapolator_args))

            return representation
        except TypeError:
            # TODO: Discuss what is the most suitable behaviour, either the
            # following or __str__ one.
            return '{0}()'.format(self.__class__.__name__)

    def __getitem__(self, x):
        if type(x) is slice:
            return self._range[x]
        else:
            return self._function(x)

    def __setitem__(self, x, value):
        if type(x) is slice:
            with ndarray_write(self._range):
                self._range[x] = value
        else:
            with ndarray_write(self._domain), ndarray_write(self._range):
                x = np.atleast_1d(x)
                value = np.resize(value, x.shape)

                # Matching domain, replacing existing `self.range`.
                mask = np.in1d(x, self._domain)
                x_m = x[mask]
                indexes = np.searchsorted(self._domain, x_m)
                self._range[indexes] = value[mask]

                # Non matching domain, inserting into existing `self.domain`
                # and `self.range`.
                x_nm = x[~mask]
                indexes = np.searchsorted(self._domain, x_nm)
                if indexes.size != 0:
                    self._domain = np.insert(self._domain, indexes, x_nm)
                    self._range = np.insert(self._range, indexes, value[~mask])

        self._create_function()

    def __contains__(self, x):
        return np.all(
            np.where(
                np.logical_and(x >= np.min(self._domain), x <=
                               np.max(self._domain)), True, False))

    def __eq__(self, x):
        if isinstance(x, Signal):
            if all([
                    np.array_equal(self._domain, x.domain),
                    np.array_equal(self._range, x.range),
                    self._interpolator is x.interpolator,
                    self._interpolator_args == x.interpolator_args,
                    self._extrapolator is x.extrapolator,
                    self._extrapolator_args == x.extrapolator_args
            ]):
                return True

        return False

    def __neq__(self, x):
        return not (self == x)

    def _create_function(self):
        if self._domain is not None and self._range is not None:
            with ndarray_write(self._domain), ndarray_write(self._range):
                # TODO: Providing a writeable copy of both `self.domain` and `
                # self.range` to the interpolator to avoid issue regarding
                # `MemoryView` being read-only.
                # https://mail.python.org/pipermail/cython-devel/2013-February/003384.html
                self._function = self._extrapolator(
                    self._interpolator(
                        np.copy(self._domain),
                        np.copy(self._range), **self._interpolator_args),
                    **self._extrapolator_args)
        else:

            def _undefined_function(*args, **kwargs):
                raise RuntimeError(
                    'Underlying signal interpolator function does not exists, '
                    'please ensure you defined both '
                    '"domain" and "range" variables!')

            self._function = _undefined_function

    def _fill_domain_nan(self, method='Interpolation', default=0):
        with ndarray_write(self._domain):
            self._domain = fill_nan(self._domain, method, default)
            self._create_function()

    def _fill_range_nan(self, method='Interpolation', default=0):
        with ndarray_write(self._range):
            self._range = fill_nan(self._range, method, default)
            self._create_function()

    def arithmetical_operation(self, a, operator, in_place=False):
        operator, ioperator = {
            '+': (add, iadd),
            '-': (sub, isub),
            '*': (mul, imul),
            '/': (div, idiv),
            '**': (pow, ipow)
        }[operator]

        if in_place:
            if isinstance(a, Signal):
                with ndarray_write(self._domain), ndarray_write(self._range):
                    self[self._domain] = operator(self._range, a[self._domain])

                    exclusive_or = np.setxor1d(self._domain, a.domain)
                    self[exclusive_or] = np.full(exclusive_or.shape, np.nan)
            else:
                with ndarray_write(self._range):
                    self.range = ioperator(self.range, a)

            return self
        else:
            copy = ioperator(self.copy(), a)

            return copy

    @staticmethod
    def signal_unpack_data(data=None, domain=None):
        domain_upk, range_upk = None, None
        if isinstance(data, Signal):
            domain_upk = data.domain
            range_upk = data.range
        elif (issubclass(type(data), Sequence) or
              isinstance(data, (tuple, list, np.ndarray, Iterator))):
            data = tsplit(list(data) if isinstance(data, Iterator) else data)
            assert data.ndim in (1, 2), (
                'User "data" must be a 1d or 2d array-like variable!')
            if data.ndim == 1:
                domain_upk, range_upk = np.arange(0, data.size), data
            else:
                domain_upk, range_upk = data
        elif (issubclass(type(data), Mapping) or
              isinstance(data, (dict, OrderedDict))):
            domain_upk, range_upk = tsplit(sorted(data.items()))
        elif is_pandas_installed():
            from pandas import Series

            if isinstance(data, Series):
                domain_upk = data.index.values
                range_upk = data.values

        if domain is not None and range_upk is not None:
            assert len(domain) == len(range_upk), (
                'User "domain" is not compatible with unpacked range!')
            domain_upk = domain

        return domain_upk, range_upk

    def fill_nan(self, method='Interpolation', default=0):
        self._fill_domain_nan(method, default)
        self._fill_range_nan(method, default)

        return self

    def uncertainty(self, a):
        n = closest(self._domain, a)

        return as_numeric(np.abs(a - n))
