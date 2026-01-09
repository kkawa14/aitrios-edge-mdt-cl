import pytest
from unittest.mock import MagicMock, patch

from edgemdt_cl.util.import_util import (
    is_compatible, 
    validate_installed_libraries, 
    RequirementError
)


class TestImportLibsCheck:

    TEST_CASES = [
        ### torch case
        (['torch'], {'torch': '2.3.0'}, True, None),
        (['torch>=2.3'], {'torch': '2.3.0'}, True, None),
        (['torch>=2.3', 'torchvision>=0.18'], {'torch': '2.3.0', 'torchvision': '0.18.0'}, True, None),
        (['torch>=2.3', 'torchvision>=0.18'], {'torch': '2.3.0a0+dev0', 'torchvision': '0.18.0.dev2+a0'}, True, None),
        (['torch>=2.3'], {}, False, "\nRequired library 'torch' is not installed."),
        (['torch>=2.3'], {'torch': '2.2.0'}, False, "\nRequired 'torch' version >=2.3, installed version 2.2.0."),
        (['torch>=2.3'], {'torch': '2.2.0a0+dev0'}, False, "\nRequired 'torch' version >=2.3, installed version 2.2.0a0+dev0."),
        (['torch>=2.3', 'torchvision>=0.18'], {'torch': '2.3.0'}, False, "\nRequired library 'torchvision' is not installed."),
        (['torch>=2.3', 'torchvision>=0.18'], {'torch': '2.2.0'}, False, \
                "\nRequired 'torch' version >=2.3, installed version 2.2.0.\nRequired library 'torchvision' is not installed."),
        (['torch>=2.3,<3.0'], {'torch': '2.5.0'}, True, None),
        (['torch>=2.3,<3.0'], {'torch': '2.5.0a0+dev0'}, True, None),
        (['torch>=2.3,<3.0'], {'torch': '3.0.0'}, False, "\nRequired 'torch' version <3.0,>=2.3, installed version 3.0.0."),
        (['torch>=2.3,<3.0'], {'torch': '2.2.0+dev1'}, False, "\nRequired 'torch' version <3.0,>=2.3, installed version 2.2.0+dev1."),

        ### tensorflow case
        (['tensorflow'], {'tensorflow': '2.15.0'}, True, None),
        (['tensorflow>=2.15'], {'tensorflow': '2.15.0'}, True, None),
        (['tensorflow>=2.15'], {}, False, "\nRequired library 'tensorflow' is not installed."),
        (['tensorflow>=2.15'], {'tensorflow': '2.10.0'}, False, "\nRequired 'tensorflow' version >=2.15, installed version 2.10.0."),
        (['tensorflow>=2.15'], {'tensorflow': '2.10.0a0+dev0'}, False, "\nRequired 'tensorflow' version >=2.15, installed version 2.10.0a0+dev0."),
        (['tensorflow>=2.14,<2.16'], {'tensorflow': '2.14.0'}, True, None),
        (['tensorflow>=2.14,<2.16'], {'tensorflow': '2.14.0a0+dev0'}, True, None),
        (['tensorflow>=2.14,<2.16'], {'tensorflow': '1.10.0'}, False, "\nRequired 'tensorflow' version <2.16,>=2.14, installed version 1.10.0."),
        (['tensorflow>=2.14,<2.16'], {'tensorflow': '1.10.0+dev1'}, False, "\nRequired 'tensorflow' version <2.16,>=2.14, installed version 1.10.0+dev1."),

        ### onnx case
        (['onnx'], {'onnx': '1.17.0'}, True, None),
        (['onnx>=1.14'], {'onnx': '1.17.0'}, True, None),
        (['onnx>=1.14', 'onnxruntime>=1.15'], {'onnx': '1.17.0', 'onnxruntime': '1.18.0'}, True, None),
        (['onnx>=1.14', 'onnxruntime>=1.15'], {'onnx': '1.17.0a0+dev0', 'onnxruntime': '1.18.0dev2+a0'}, True, None),
        (['onnx>=1.14'], {}, False, "\nRequired library 'onnx' is not installed."),
        (['onnx>=1.14'], {'onnx': '1.0.0'}, False, "\nRequired 'onnx' version >=1.14, installed version 1.0.0."),
        (['onnx>=1.14'], {'onnx': '1.0.0a0+dev0'}, False, "\nRequired 'onnx' version >=1.14, installed version 1.0.0a0+dev0."),
        (['onnx>=1.14', 'onnxruntime>=1.15', 'onnxruntime_extensions>=0.8.0'], {'onnx': '1.17.0', 'onnxruntime_extensions': '0.9.0'}, False, "\nRequired library 'onnxruntime' is not installed."),
        (['onnx>=1.14', 'onnxruntime>=1.15', 'onnxruntime_extensions>=0.8.0'], {'onnx': '1.0.0'}, False, \
                "\nRequired 'onnx' version >=1.14, installed version 1.0.0.\nRequired library 'onnxruntime' is not installed.\nRequired library 'onnxruntime_extensions' is not installed."),
        (['onnx>=1.14,<1.18'], {'onnx': '1.17.0'}, True, None),
        (['onnx>=1.14,<1.18'], {'onnx': '1.17.0a0+dev0'}, True, None),
        (['onnx>=1.14,<1.18'], {'onnx': '1.3.0'}, False, "\nRequired 'onnx' version <1.18,>=1.14, installed version 1.3.0."),
        (['onnx>=1.14,<1.18'], {'onnx': '1.3.0+dev1'}, False, "\nRequired 'onnx' version <1.18,>=1.14, installed version 1.3.0+dev1."),
    ]

    @staticmethod
    def _create_mock_import_module(mock_modules):
        def mock_import_module(name):
            if name not in mock_modules:
                raise ImportError(f"No module named '{name}'")
        
            module = MagicMock()
            version = mock_modules[name]

            if version is not None:
                module.__version__ = version
            else:
                del module.__version__
            return module
        return mock_import_module

    @pytest.mark.parametrize(("requirements", "mock_modules", "expected"), 
        [(req, mock, expected) for req, mock, expected, _ in TEST_CASES]
    )
    def test_is_compatible(self, requirements, mock_modules, expected):
        mock_import_module = self._create_mock_import_module(mock_modules)
        
        with patch('edgemdt_cl.util.import_util.importlib.import_module', side_effect=mock_import_module):
            assert is_compatible(requirements) == expected

    @pytest.mark.parametrize(("requirements", "mock_modules", "expected"),
        [(req, mock, expected) for req, mock, _, expected in TEST_CASES]
    )
    def test_validate_installed_libraries(self, requirements, mock_modules, expected):
        mock_import_module = self._create_mock_import_module(mock_modules)
        
        with patch('edgemdt_cl.util.import_util.importlib.import_module', side_effect=mock_import_module):
            if expected is not None:
                with pytest.raises(RequirementError) as e:
                    validate_installed_libraries(requirements)
                assert str(e.value) == expected
            else:
                validate_installed_libraries(requirements)
