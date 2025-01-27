import random


def random01biased():
    p = 0.7
    return 0 if random.random() < p else 1


def getFairCoin():
    while True:
        x1 = random01biased()
        x2 = random01biased()
        # p * (1 - p) == (1 - p) * p
        if x1 != x2:
            return x1


def random06uniform():
    while True:
        num = (getFairCoin() << 2) | (getFairCoin() << 1) | getFairCoin()
        if num <= 6:
            return num
