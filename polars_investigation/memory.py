import resource


def memory_consumption_in_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
