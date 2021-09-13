class LossFactory:
    """
    LossFactory builds and returns a loss objects

    loss objects share the following properties:  PLACEHOLDER
    """

    def __init__(self):
        self._creators = {}

    def register_format(self, loss_name, creator):
        self._creators[loss_name] = creator

    def get_loss(self, loss_name, **loss_params):
        creator = self._creators.get(loss_name)
        loss_params["loss_name"] = loss_name
        if creator is None:
            return Loss(**loss_params)
        return creator(**loss_params)


def get_loss_creator(**losses_dict):
    factory = LossFactory()
    for k, v in losses_dict.items():
        factory.register_format(k, v)

    return factory


class Loss(object):
    """
    Generic Loss object, not intended to be used except as abstract parent
    """

    def __init__(self, **loss_params):
        name = loss_params.get("loss_name")
        self.loss_name = "loss" if name is None else name
        self._loss_params = loss_params
        self.value = None
        self._information_string = """This loss object reports some information about inputs given.
        If inheriting this parent object, remember to override this information with something helpful"""

    def __repr__(self):
        return str(
            {
                "type": type(self),
                "loss_name": self.loss_name,
                "_loss_params": self._loss_params,
                "_information_string": self._information_string,
            }
        )

    def __str__(self):
        s = (
            f"""{type(self).__name__}:\n"""
            f"""loss_name:\n{self.loss_name}\n"""
            f"""_loss_params: {self._loss_params}\n"""
            f"""_information_string: {self._information_string}"""
        )
        return s

    def compute(self, *loss_args, **loss_kwargs):
        """
        Abstract compute function
        """
        return self.value

    def score(self):
        """
        Returns self.value

        Remember to overwrite this when your value is not a loss
        """
        return self.value

    def info(self):
        return self._information_string


class CombinedLoss(Loss):
    """
    Combines losses based on some rule

    Abstract function, losses are generally expected to have a value before this is invoked
    """

    def __init__(self, *losses, **combine_params):
        name = combine_params.get("loss_name")
        self.loss_name = "combined_loss" if name is None else name
        self.losses = losses
        self.value = None
        self._loss_params = combine_params
        self._information_string = """This loss object reports a combined score from different losses.
        It also saves each sub-loss score individually for reporting. Takes fully instantiated losses as arguments.
        If inheriting this parent object, remember to override this information with something helpful"""

    def get_base_values(self):
        """
        Returns the loss name,value pair for each sub-loss making up this combined one

        Don't overwrite this abstract func without good reason, this is the key part of the schema
        """
        all_vals = {}
        for loss in self.losses:
            try:
                vals_dict = loss.get_base_values()
                for k, v in vals_dict.items():
                    all_vals[k] = v
            except AttributeError:
                all_vals[loss.loss_name] = loss.value
        return all_vals
