"""
Dynamic architecture loading and management system for the training pipeline.
"""

import os
import sys
import importlib
import importlib.util
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
import json
import shutil
import traceback
from dataclasses import dataclass

from napari.utils.notifications import show_error, show_info, show_warning


@dataclass
class ArchitectureInfo:
    name: str
    description: str
    path: Path
    config_parameters: Dict[str, Any]
    default_config: Dict[str, Any]
    train_data_type: str
    dependencies: List[str]


class ArchitectureManager:
    """
    Manages discovery, loading, and validation of custom architectures.
    
    Architecture Directory Structure:
    architectures/
    └── YourArchitecture/
        ├── __init__.py          # Must contain: from <any module> import YourClass; __all__ = ['YourClass']
        └── requirements.txt     # Optional dependencies
    
    The __init__.py file MUST define __all__ with exactly one class name that will be
    imported and used as the architecture class.
    
    Example __init__.py:
        from .main import KNNClassifierArchitecture
        __all__ = ['KNNClassifierArchitecture']
    """
    
    def __init__(self, architectures_dir):
        self.architectures_dir = Path(architectures_dir)
        self.architectures_dir.mkdir(exist_ok=True)
        self.discovered_architectures = {}
        self.loaded_architectures = {}

    def discover_architectures(self):

        self.discovered_architectures.clear()
        
        if not self.architectures_dir.exists():
            return {}
            
        for arch_dir in self.architectures_dir.iterdir():
            if arch_dir.is_dir() and not arch_dir.name.startswith('.'):
                try:
                    arch_info = self._parse_architecture_dir(arch_dir)
                    if arch_info:
                        self.discovered_architectures[arch_info.name] = arch_info
                except Exception as e:
                    show_warning(f"Failed to parse architecture in {arch_dir.name}: {str(e)}")
                    
        return self.discovered_architectures
    
    def get_architecture_info(self, arch_name: str):
        """
        Get architecture information by name.
        """
        return self.discovered_architectures.get(arch_name, None)
    
    def get_discovered_architectures(self):
        """
        Returns a dictionary of discovered architectures.
        """
        return self.discovered_architectures
    
    def _import_arch_class(self, arch_dir, reload_module=False):
        """
        Import the architecture class from main.py in the given directory.
        
        Uses __all__ list in __init__.py to determine which class to import.
        The __init__.py file should contain __all__ = ['ArchitectureClassName']
        """
        parent_dir = str(arch_dir.parent)
        arch_package_name = arch_dir.name
        
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            path_added = True
        else:
            path_added = False
        
        def cleanup(delete_from_sys_modules=False):
            if path_added and parent_dir in sys.path:
                sys.path.remove(parent_dir)
            if delete_from_sys_modules:
                to_delete = [key for key in sys.modules if key == arch_package_name or key.startswith(arch_package_name + ".")]
                for key in to_delete:
                    del sys.modules[key]

        try:
            arch_package = importlib.import_module(arch_package_name)
            
            if reload_module:
                importlib.reload(arch_package)
            
            if not hasattr(arch_package, '__all__') or not arch_package.__all__:
                show_error(f"Architecture {arch_dir.name} must define __all__ in its __init__.py file")
                return None, cleanup
            
            # For now we support exactly one class in __all__.
            # TODO: Explore supporting multiple classes in the future if needed
            if len(arch_package.__all__) != 1:
                show_error(f"Architecture {arch_dir.name} must specify exactly one class in __all__, found: {arch_package.__all__}")
                return None, cleanup
            
            arch_class_name = arch_package.__all__[0]
            
            if not hasattr(arch_package, arch_class_name):
                show_error(f"Architecture {arch_dir.name} specifies '{arch_class_name}' in __all__ but class not found")
                return None, cleanup
            
            arch_class = getattr(arch_package, arch_class_name)
            
            if not isinstance(arch_class, type):
                show_error(f"Architecture {arch_dir.name}: '{arch_class_name}' is not a class")
                return None, cleanup
            
            print(f"Successfully imported architecture class: {arch_class.__name__}")
            return arch_class, cleanup
            
        except Exception as e:
            show_error(f"Failed to import architecture class from {arch_dir.name}: {str(e)}")
            cleanup()
            return None, None
    
    def _parse_architecture_dir(self, arch_dir, delete_from_sys_modules=False):
        """
        Parses an architecture directory to extract it as a Python module.
        """

        if not arch_dir.is_dir():
            show_error(f"Architecture path {arch_dir} is not a directory")
            return None
        
        init_file = arch_dir / "__init__.py"
        if not init_file.exists():
            show_error(f"Architecture {arch_dir.name} must contain an __init__.py file")
            return None
            
        requirements_file = arch_dir / "requirements.txt"

        dependencies = []
        if requirements_file.exists():
            dependencies = requirements_file.read_text().strip().split('\n')
            dependencies = [dep.strip() for dep in dependencies if dep.strip()]
        
        arch_class, cleanup = self._import_arch_class(arch_dir)
        print(f"Architecture class: {arch_class}")
        if arch_class is None:
            return None
        
        try:
            arch_name = arch_class.architecture_name
            arch_description = arch_class.architecture_description
            config_parameters = arch_class.config_parameters
            train_data_type = arch_class.train_data_type
            default_config = arch_class.default_config

            config_schema_valid, config_schema_errors = self._validate_config_schema(config_parameters)
            if not config_schema_valid:
                show_error(f"Architecture {arch_name} has invalid config schema: {', '.join(config_schema_errors)}")
                return None

            default_config_valid, default_config_errors = self.validate_config(config_parameters, default_config)
            if not default_config_valid:
                show_error(f"Architecture {arch_name} has invalid default config: {', '.join(default_config_errors)}")
                return None
            
            return ArchitectureInfo(
                name=arch_name,
                description=arch_description,
                path=arch_dir,
                config_parameters=config_parameters,
                default_config=default_config,
                train_data_type=train_data_type,
                dependencies=list(set(dependencies)),
            )
            
        except Exception as e:
            show_error(f"Failed to load architecture {arch_dir.name}: {str(e)}")
            delete_from_sys_modules = True
            return None
        finally:
            cleanup(delete_from_sys_modules=delete_from_sys_modules)

    def install_dependencies(self, arch_name):
        """
        Install dependencies for a given architecture.
        """
        if arch_name not in self.discovered_architectures:
            show_error(f"Architecture {arch_name} not found")
            return False
            
        arch_info = self.discovered_architectures[arch_name]
        
        if not arch_info.dependencies:
            return True
            
        try:
            for dependency in arch_info.dependencies:
                if dependency.strip():
                    show_info(f"Installing {dependency}...")
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", dependency
                    ], capture_output=True, text=True, check=True)
                    
            show_info(f"Successfully installed dependencies for {arch_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            show_error(f"Failed to install dependencies for {arch_name}: {e.stderr}")
            return False
        except Exception as e:
            show_error(f"Unexpected error installing dependencies: {str(e)}")
            return False
    
    def load_architecture(self, arch_name, config):
        """
        Load an architecture by name.
        """
        if arch_name not in self.discovered_architectures:
            show_error(f"Architecture {arch_name} not found")
            return None
            
        arch_info = self.discovered_architectures[arch_name]
        
        if arch_info.dependencies and not self._check_dependencies_installed(arch_info.dependencies):
            if not self.install_dependencies(arch_name):
                return None
        
        try:
            arch_class, cleanup = self._import_arch_class(arch_info.path, reload_module=True)
            if arch_class is None:
                return None
            
            instance = arch_class(config)
            
            self.loaded_architectures[arch_name] = instance
            show_info(f"Successfully loaded architecture: {arch_name}")
            return instance
            
        except Exception as e:
            show_error(f"Failed to load architecture {arch_name}: {str(e)}")
            traceback.print_exc()
            return None
        finally:
            if cleanup:
                cleanup()
    
    def _check_dependencies_installed(self, dependencies):

        for dependency in dependencies:
            if not dependency.strip():
                continue
                
            package_name = dependency.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
            
            try:
                importlib.import_module(package_name)
            except ImportError:
                return False
                
        return True
    
    @staticmethod
    def _validate_config_schema(params_schema):
        """
        Validate that provided configuration matches the expected schema.
        """
        errors = []
        
        for param_name, param_spec in params_schema.items():
            if isinstance(param_spec, str):
                if param_spec not in ["float", "int", "str", "bool"]:
                    errors.append(f"Invalid type for parameter {param_name}: {param_spec}")
            elif isinstance(param_spec, list):
                if not all(isinstance(opt, str) for opt in param_spec):
                    errors.append(f"Selector options for {param_name} must be strings")
                    
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_config(params_schema, config):
        """
        Validate that provided configuration matches the expected schema.
        """
        errors = []
        
        for param_name, param_spec in params_schema.items():
            if param_name not in config:
                errors.append(f"Missing required parameter: {param_name}")
                continue
                
            value = config[param_name]
            
            if isinstance(param_spec, str):
                if param_spec == "float":
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        errors.append(f"Parameter {param_name} must be a float")
                elif param_spec == "int":
                    try:
                        int(value)
                    except (ValueError, TypeError):
                        errors.append(f"Parameter {param_name} must be an integer")
                elif param_spec == "str":
                    if not isinstance(value, str):
                        errors.append(f"Parameter {param_name} must be a string")
                elif param_spec == "bool":
                    if not isinstance(value, bool):
                        errors.append(f"Parameter {param_name} must be a boolean")
            elif isinstance(param_spec, list):
                if value not in param_spec:
                    errors.append(f"Parameter {param_name} must be one of: {param_spec}")
                    
        return len(errors) == 0, errors
