import os


class Random:

    @staticmethod
    def urandom_float():
        random_bytes = os.urandom(8)
        random_int = int.from_bytes(random_bytes, byteorder='big')
        return random_int / (1 << 64)
