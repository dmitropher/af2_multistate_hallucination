class ScoreContainer(object):
    """
    Container for all scores
    """
    def __init__(self):
        ""
        self.scores = []



class Score(object):
    ""
    def __init__(self):
        ""

def scores_from_loss(*args,**kwargs):
    """
    Extract the score(s) from a Loss object

    This detects if the Loss is a combination, and decomposes
    it into individual terms if necessary
    """
