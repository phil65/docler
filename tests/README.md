# Testing

## Running Tests

```bash
uv run pytest                         # Run regular tests (no integration tests)
uv run pytest --integration          # Run all tests including integration tests
uv run pytest -m integration         # Run only integration tests
```

## Snapshot Testing

```bash
# Update snapshots
uv run pytest --snapshot-update --integration    # Update all snapshots
uv run pytest --snapshot-update -m integration   # Update only integration test snapshots

# Review changes
uv run pytest --snapshot-review                  # Interactive review of snapshot changes
```

## Test Types

- Regular tests: Unit tests, mock-based tests
- Integration tests: Tests that interact with external services or require sample files
  - Marked with `@pytest.mark.integration`
  - Disabled by default, enable with `--integration` flag
