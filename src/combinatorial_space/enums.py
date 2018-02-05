from enum import Enum


class ClusterAnswer(Enum):
    ACTIVE = 1
    NOT_ACTIVE = 0
    MODIFY = 2
    NOT_MODIFY = 3


class PredictEnum(Enum):
    INACTIVE_POINTS = 0
    ACCEPT = 1


class LearnEnum(Enum):
    LEARN = 0
    SLEEP = 1
    BAD_CODES = 1


class PointPredictAnswer(Enum):
    NOT_ACTIVE = 0
    ACTIVE = 1
    NO_CLUSTERS = 2