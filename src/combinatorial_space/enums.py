from enum import Enum


class CLUSTER(Enum):
    ACTIVE = 1
    NOT_ACTIVE = 0
    MODIFY = 2
    NOT_MODIFY = 3


class POINT_PREDICT(Enum):
    NOT_ACTIVE = 0
    ACTIVE = 1
    NO_CLUSTERS = 2


class MINICOLUMN_LEARNING(Enum):
    INACTIVE_POINTS = 0
    ACCEPT = 1


class MINICOLUMN(Enum):
    LEARN = 0
    SLEEP = 1
    BAD_CODES = 2
