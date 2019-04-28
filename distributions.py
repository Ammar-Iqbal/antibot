def positive(func, args):
    def positive_impl(dist, func, args):
        if dist[dist < 0].size == 0:
            return dist
        else:
            args = args[:2] + (dist[dist < 0].size,)
            dist[dist < 0] = func(*args)
            return positive_impl(dist, func, args)

    return positive_impl(func(*args), func, args)
