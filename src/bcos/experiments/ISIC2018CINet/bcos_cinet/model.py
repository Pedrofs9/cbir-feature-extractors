# Project Imports
from bcos.models.cinet import bcos_cinet



__all__ = ["get_model"]



# Function: get_model
def get_model(model_config):
   
    # Extract arguments
    args = model_config["args"]

    # Buil model
    model = bcos_cinet(**args)

    return model
