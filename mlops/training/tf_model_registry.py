import os
import importlib
import pkgutil

# Global model registry
MODEL_REGISTRY = {}

def register_model(name: str, description: str):
    """
    Decorator to register a model function.
    """
    def decorator(fn):
        fn.description = description
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator

def _import_all_model_modules():
    """
    Dynamically imports all modules in the 'training.models' subpackage.
    """
    # Absolute path to the models/ folder
    package_dir = os.path.join(os.path.dirname(__file__), "models")
    package_name = "mlops.training.models"

    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        full_module_name = f"{package_name}.{module_name}"
        importlib.import_module(full_module_name)

# Autoload on import
_import_all_model_modules()