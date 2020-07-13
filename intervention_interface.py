from intervention_model.model import InterventionModel
from intervention_model.model import ComputationGraphNode as Node
from torch_equality import TorchEqualityModel

import torch

class TorchEqualityIntervention(InterventionModel):
    def __init__(self, model):
        assert isinstance(model, TorchEqualityModel)
        self.model = model
        self.update_vector_cache()

    def update_vector_cache(self):
        """ Here the user defines how to map strings to acceses vectors in the model """
        self.vector_cache = {
            "hidden_vec": getattr(self.model.module, "hidden_vec", None)
        }

    def get_from_cache(self, name):
        """ Obtain the current values of an internal vector in the cache """
        return self.vector_cache[name]

    def set_to_cache(self, name, value):
        """ Set the values of an internal vector in cache"""
        self.vector_cache[name] = value

    def run(self, inputs):
        """ Run the model on a set of inputs, we expect this to update *all*
        internal vectors of the model according to the new input"""
        outputs = self.model.predict_one(inputs)
        self.update_vector_cache()
        return outputs

    def fix_and_run(self, names, inputs):
        """ Run the model on a set of inputs, but hold a set of internal vectors
        constant. This set is given by the list of strings in `names`"""
        inputs = inputs.float().to(self.model.device)
        mod = self.model.module
        with torch.no_grad():
            linear_out = mod.linear(inputs)
            if "hidden_vec" in names:
                mod.hidden_vec = self.vector_cache["hidden_vec"]
            else:
                mod.hidden_vec = mod.activation(linear_out)
            logits = mod.output(mod.hidden_vec)
            scores = mod.sigmoid(logits)
            return [1 if z >= 0.5 else 0 for z in scores]

    def get_causal_ordering(self, name):
        """ Get a data structure of what downstream vectors are affected by
        one internal vector designated by `name`"""
        pass
