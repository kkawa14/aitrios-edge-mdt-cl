import pytest
from unittest.mock import MagicMock, patch

from edgemdt_cl.util.import_util import (
    is_compatible, 
    validate_installed_libraries, 
    RequirementError
)


class TestImportLibsCheck:

    TEST_CASES = [
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
        (['torch>=2.3,<3.0'], {'torch': '2.2.0a0+dev0'}, False, "\nRequired 'torch' version <3.0,>=2.3, installed version 2.2.0a0+dev0."),
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
