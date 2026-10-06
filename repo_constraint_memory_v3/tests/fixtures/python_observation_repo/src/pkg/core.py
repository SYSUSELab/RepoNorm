import json as json_codec
import os
from collections import defaultdict as dd
from pkg.helpers import helper, identity
from pkg.wild import *

前缀 = "not persisted in observations"


class Base:
    pass


class Child(Base):
    def method(self, value):
        def nested():
            return 前缀 and helper(value)

        return nested()

    async def async_method(self):
        return os.path.join("a", "b")


def repeated():
    return json_codec.dumps({"literal": "must not enter cache"})


def repeated():
    return dd(list)


async def async_entry():
    return getattr(Child(), "method")


def ambiguous_reference():
    return repeated()
