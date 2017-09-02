#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi Signal
============

Defines the class implementing support for multiple continuous signal data
representation:

-   :class:`MultiSignal`
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import Iterator, Mapping, OrderedDict, Sequence

# Python 3 compatibility.
try:
    from operator import div, idiv
except ImportError:
    from operator import truediv, itruediv

    div = truediv
    idiv = itruediv

from colour.continuous import AbstractContinuousFunction, Signal
from colour.utilities import first_item, is_pandas_installed, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['MultiSignal']


class MultiSignal(AbstractContinuousFunction):
    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(MultiSignal, self).__init__(kwargs.get('name'))

        self._signals = self.multi_signal_unpack_data(data, domain, labels)

    @property
    def domain(self):
        if self._signals:
            return first_item(self._signals.values()).domain

    @domain.setter
    def domain(self, value):
        if value is not None:
            for signal in self._signals.values():
                signal.domain = value

    @property
    def range(self):
        if self._signals:
            return tstack([signal.range for signal in self._signals.values()])

    @range.setter
    def range(self, value):
        if value is not None:
            for signal in self._signals.values():
                signal.range = value

    @property
    def interpolator(self):
        if self._signals:
            return first_item(self._signals.values()).interpolator

    @interpolator.setter
    def interpolator(self, value):
        if value is not None:
            for signal in self._signals.values():
                signal.interpolator = value

    @property
    def interpolator_args(self):
        if self._signals:
            return first_item(self._signals.values()).interpolator_args

    @interpolator_args.setter
    def interpolator_args(self, value):
        if value is not None:
            for signal in self._signals.values():
                signal.interpolator_args = value

    @property
    def extrapolator(self):
        if self._signals:
            return first_item(self._signals.values()).extrapolator

    @extrapolator.setter
    def extrapolator(self, value):
        if value is not None:
            for signal in self._signals.values():
                signal.extrapolator = value

    @property
    def extrapolator_args(self):
        if self._signals:
            return first_item(self._signals.values()).extrapolator_args

    @extrapolator_args.setter
    def extrapolator_args(self, value):
        if value is not None:
            for signal in self._signals.values():
                signal.extrapolator_args = value

    @property
    def function(self):
        if self._signals:
            return first_item(self._signals.values()).function

    @function.setter
    def function(self, value):
        raise AttributeError(
            '"{0}" attribute is read only!'.format('function'))

    @property
    def signals(self):
        return self._signals

    @signals.setter
    def signals(self, value):
        if value is not None:
            self._signals = self.multi_signal_unpack_data(value)

    @property
    def labels(self):
        if self._signals:
            return list(self._signals.keys())

    @labels.setter
    def labels(self, value):
        if value is not None:
            assert len(value) == len(self._signals), (
                '"labels" length does not match "signals" length!')
            self._signals = OrderedDict(
                [(value[i], signal)
                 for i, (_key, signal) in enumerate(self._signals.items())])

    def __str__(self):
        try:
            return str(np.hstack((self.domain[:, np.newaxis], self.range)))
        except TypeError:
            return super(MultiSignal, self).__str__()

    def __repr__(self):
        try:
            representation = repr(
                np.hstack((self.domain[:, np.newaxis], self.range)))
            representation = representation.replace('array',
                                                    self.__class__.__name__)
            representation = representation.replace('       [', '{0}['.format(
                ' ' * (len(self.__class__.__name__) + 2)))
            representation = ('{0},\n'
                              '{1}labels={2},\n'
                              '{1}interpolator={3},\n'
                              '{1}interpolator_args={4},\n'
                              '{1}extrapolator={5},\n'
                              '{1}extrapolator_args={6})').format(
                                  representation[:-1],
                                  ' ' * (len(self.__class__.__name__) + 1),
                                  repr(self.labels), self.interpolator.__name__
                                  if self.interpolator is not None else
                                  self.interpolator,
                                  repr(self.interpolator_args),
                                  self.extrapolator.__name__
                                  if self.extrapolator is not None else
                                  self.extrapolator,
                                  repr(self.extrapolator_args))

            return representation
        except TypeError:
            # TODO: Discuss what is the most suitable behaviour, either the
            # following or __str__ one.
            return '{0}()'.format(self.__class__.__name__)

    def __getitem__(self, x):
        if self._signals:
            return tstack([signal[x] for signal in self._signals.values()])
        else:
            raise RuntimeError('No underlying "Signal" defined!')

    def __setitem__(self, x, value):
        for signal in self._signals.values():
            signal[x] = value

    def __contains__(self, x):
        if self._signals:
            return x in first_item(self._signals.values())
        else:
            raise RuntimeError('No underlying "Signal" defined!')

    def __eq__(self, x):
        if isinstance(x, MultiSignal):
            if all([
                    np.array_equal(self.domain, x.domain),
                    np.array_equal(self.range, x.range),
                    self.interpolator is x.interpolator,
                    self.interpolator_args == x.interpolator_args,
                    self.extrapolator is x.extrapolator,
                    self.extrapolator_args == x.extrapolator_args
            ]):
                return True

        return False

    def __neq__(self, x):
        return not (self == x)

    def arithmetical_operation(self, a, operator, in_place=False):
        multi_signal = self if in_place else self.copy()

        if isinstance(a, MultiSignal):
            assert len(self.signals) == len(a.signals), (
                '"MultiSignal" operands must have same count of '
                'underlying "Signal" components!')
            for signal_a, signal_b in zip(multi_signal.signals.values(),
                                          a.signals.values()):
                signal_a.arithmetical_operation(signal_b, operator, True)
        else:
            for signal in multi_signal.signals.values():
                signal.arithmetical_operation(a, operator, True)

        return multi_signal


    @staticmethod
    def multi_signal_unpack_data(data=None, domain=None, labels=None):
        domain_upk, range_upk, signals = None, None, None
        signals = OrderedDict()
        if isinstance(data, MultiSignal):
            signals = data.signals
        elif (issubclass(type(data), Sequence) or
              isinstance(data, (tuple, list, np.ndarray, Iterator))):
            data = tsplit(list(data) if isinstance(data, Iterator) else data)
            assert data.ndim in (1, 2), (
                'User "data" must be a 1d or 2d array-like variable!')
            if data.ndim == 1:
                signals[0] = Signal(data)
            else:
                domain_upk, range_upk = ((data[0], data[1:])
                                         if domain is None else (domain, data))
                for i, range_upk_c in enumerate(range_upk):
                    signals[i] = Signal(range_upk_c, domain_upk)
        elif (issubclass(type(data), Mapping) or
              isinstance(data, (dict, OrderedDict))):
            domain_upk, range_upk = tsplit(sorted(data.items()))
            for i, range_upk in enumerate(tsplit(range_upk)):
                signals[i] = Signal(range_upk, domain_upk)
        elif is_pandas_installed():
            from pandas import DataFrame, Series

            if isinstance(data, Series):
                signals[0] = Signal(data)
            elif isinstance(data, DataFrame):
                # Check order consistency.
                domain_upk = data.index.values
                signals = OrderedDict(((label, Signal(
                    data[label], domain_upk, name=label)) for label in data))

        if domain is not None and signals is not None:
            for signal in signals.values():
                assert len(domain) == len(signal.domain), (
                    'User "domain" is not compatible with unpacked signals!')
                signal.domain = domain

        if labels is not None and signals is not None:
            assert len(labels) == len(signals), (
                'User "labels" is not compatible with unpacked signals!')
            signals = OrderedDict(
                [(labels[i], signal)
                 for i, (_key, signal) in enumerate(signals.items())])

        return signals

    def fill_nan(self, method='Interpolation', default=0):
        for signal in self._signals.values():
            signal.fill_nan(method, default)

        return self

    def uncertainty(self, a):
        if self._signals:
            return first_item(self._signals.values()).uncertainty(a)
