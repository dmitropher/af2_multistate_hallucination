class ScoreContainer(object):
    """
    Container for all scores
    """
    def __init__(self,*scores):
        ""
        self._scores_dict = {score.name:score.value for score in scores}

    def add_scores(self,*scores,overwrite=True):
        """
        add in scores: overwrites scores with same name by default
        """
        if overwrite:

            for score in scores:
                self._scores_dict[score.name] = score.value
        else:
            for score in scores:
                if not (score.name in self._scores_dict.keys()):
                    self._scores_dict[score.name] = score.value

    def add_score(self,score,overwrite=True):
        """
        add single score: overwrites score with same name by default
        """
        if overwrite:
            self._scores_dict[score.name] = score.value
        else:
            if not (score.name in self._scores_dict.keys()):
                self._scores_dict[score.name] = score.value

    def get_score(self,name):
        return self._scores_dict[name]

    def get_keys(self):
        return self._scores_dict.keys()



class Score(object):
    ""
    def __init__(self,name,value=None):
        ""
        self.name = name
        self.value = value

def scores_from_loss(loss_obj):
    """
    Extract the score(s) from a Loss object

    This detects if the Loss is a combination, and decomposes
    it into individual terms if necessary
    """
    all_scores = {}
    all_scores[loss_obj.loss_name] = loss_obj.score()
    sub_values = loss_obj.get_base_values()
    for k,v in sub_values.items():
        all_scores[k] = v
    out_scores = [Score(k,value=v) for k,v in all_scores.items()]
    return out_scores
