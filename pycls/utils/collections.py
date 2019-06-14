#!/usr/bin/env python3

"""Attribute dict structure used for config options (adapted from Detectron)."""


class AttrDict(dict):

    def __getattr__(self, name):
        if name not in self:
            raise AttributeError(name)
        return self[name]

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value
