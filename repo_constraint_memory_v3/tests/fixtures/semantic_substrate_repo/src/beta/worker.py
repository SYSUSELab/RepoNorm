from alpha.core import Resource


class Worker(Resource):
    enabled = True

    def run(self):
        return self.compute("worker")


def make_worker():
    return Worker()
