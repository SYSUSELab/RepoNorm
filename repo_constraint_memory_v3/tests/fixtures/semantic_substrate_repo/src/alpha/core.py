import functools

MODULE_FLAG = True


def marker(name):
    def decorate(value):
        return value

    return decorate


@marker("component")
class Resource:
    category = "generic"

    @functools.lru_cache()
    def compute(self, value):
        prefix = "semantic substrate"

        def nested(extra):
            return f"{prefix}:{value}:{extra}"

        return nested("done")

    async def close(self):
        return None


@marker("entry")
async def entrypoint(value):
    resource = Resource()
    return resource.compute(value)


AFTER_DEFINITIONS = "module-owned"
