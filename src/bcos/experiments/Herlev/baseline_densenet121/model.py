# Project Imports
import bcos.models.densenet as densenet



__all__ = ["get_model"]



# Function: get_arch_builder
def get_arch_builder(arch_name: str):
    arch_builder = None

    if arch_name.startswith("densenet"):
        arch_builder = getattr(densenet, arch_name)
    elif arch_name.startswith("baseline"):
        arch_builder = getattr(densenet, arch_name)

    assert arch_builder is not None
    return arch_builder



# Function: get_model
def get_model(model_config):
   
    # Extract arguments
    arch_name = model_config["name"]
    args = model_config["args"]

    # Buil model
    model = get_arch_builder(arch_name)(**args)

    return model
