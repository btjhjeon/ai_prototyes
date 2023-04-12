import string
import random


def get_random_string(length=20):
    # choose from all lowercase letter
    letters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))
