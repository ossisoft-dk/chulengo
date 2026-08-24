# Chulengo Test Suite

A comprehensive pytest-based test suite for the Chulengo llama.cpp wrapper.

## Overview

This test suite provides complete coverage of the Chulengo codebase, testing all major functionality including:

- Settings loading and saving
- HuggingFace cache path detection
- Model family detection
- Default settings retrieval
- Command building for llama.cpp
- CLI commands (ls, show, create, update, delete, serve)

## Installation

### Install Test Dependencies

```bash
# Install from requirements file
pip install -r requirements-test.txt

# Or install manually
pip install pytest pytest-mock pyyaml
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run Specific Test Class

```bash
pytest tests/test_chulengo.py::TestDetectModelFamily -v
```

### Run Specific Test

```bash
pytest tests/test_chulengo.py::test_detects_qwen3 -v
```

### Run Tests with Coverage Report

```bash
pip install pytest-cov
pytest tests/ --cov=chulengo --cov-report=html
```

## Test Structure

```
tests/
├── __init__.py           # Package marker
├── conftest.py           # Shared pytest fixtures
├── test_chulengo.py      # Main test file (54 tests)
└── README.md             # This file
```

## Test Classes

| Class                             | Tests | Description                           |
| --------------------------------- | ----- | ------------------------------------- |
| `TestLoadSettings`                | 4     | Tests for loading settings from YAML  |
| `TestSaveSettings`                | 2     | Tests for saving settings to config   |
| `TestGetHfCachePath`              | 7     | Tests for HF cache path detection     |
| `TestDetectModelFamily`           | 11    | Tests for model family detection      |
| `TestGetDefaultSettings`          | 2     | Tests for default settings retrieval  |
| `TestBuildLlamaCommand`           | 9     | Tests for command building            |
| `TestBuildLlamaCommandAdditional` | 4     | Additional command building tests     |
| `TestCmdCreate`                   | 2     | Tests for create CLI command          |
| `TestCmdDelete`                   | 2     | Tests for delete CLI command          |
| `TestCmdShow`                     | 2     | Tests for show CLI command            |
| `TestCmdUpdate`                   | 2     | Tests for update CLI command          |
| `TestCmdLs`                       | 2     | Tests for ls CLI command              |
| `TestGetModelCacheInfo`           | 1     | Tests for cache info retrieval        |
| `TestIntegration`                 | 1     | End-to-end workflow tests             |
| `TestModelFamilyCoverage`         | 2     | Coverage tests for all model families |

## Test Features

### Mocking Strategy

The test suite uses `pytest-mock` to mock:

- **File I/O**: `Path.exists()`, `Path.glob()`, file reads/writes
- **Environment variables**: `os.environ` for cache path detection
- **Subprocess calls**: `subprocess.Popen` for command execution
- **External dependencies**: HuggingFace cache paths

### Fixtures

The `conftest.py` file provides reusable fixtures:

- `default_settings_content` - YAML content for default settings
- `mock_filesystem` - Mocked filesystem operations
- `sample_model_names` - Sample model names for testing
- `sample_model_families` - Mapping of models to expected families

## Model Families Tested

All model families from `models.yaml` are tested:

- qwen3, qwancoder, qwopus
- codestral
- llama3, llama3_1, llama3_2, llama
- gemma, gemma3
- mistral, mixtral
- granite
- phi
- gpt-oss

## Continuous Integration

For CI/CD integration, add this to your workflow:

```yaml
- name: Run tests
  run: pytest tests/ -v
```

## Writing New Tests

When adding new tests:

1. Create a new test class for new functionality
2. Use descriptive test names: `test_<function>_<scenario>`
3. Mock external dependencies appropriately
4. Test both success and error cases
5. Use fixtures from `conftest.py` when possible

## Troubleshooting

### Tests fail with import errors

Ensure you're running from the project root directory:

```bash
cd /path/to/chulengo
pytest tests/
```

### Tests fail due to environment variables

Some tests modify `os.environ`. They should be isolated, but if you see failures:

```bash
pytest tests/ --forked  # Run each test in a separate process
```

### Mock issues

Ensure you're using the `mocker` fixture provided by `pytest-mock`:

```python
def test_something(self, mocker):
    mocker.patch("chulengo.load_settings", return_value={...})
```
