# Python Best Practices for Clean Code

## Naming Conventions

Python follows PEP 8 naming conventions. Here's a quick reference:

- **Variables and functions**: Use `snake_case`. Example: `user_count`, `calculate_total()`
- **Classes**: Use `PascalCase`. Example: `UserProfile`, `DatabaseConnection`
- **Constants**: Use `UPPER_SNAKE_CASE`. Example: `MAX_RETRIES`, `API_BASE_URL`
- **Private attributes**: Prefix with underscore. Example: `_internal_state`
- **"Dunder" methods**: Double underscore prefix and suffix. Example: `__init__`, `__repr__`

## Function Design

### Keep Functions Small and Focused

Each function should do one thing well. If you find yourself writing a function longer than 20-30 lines, consider breaking it into smaller helper functions.

```python
# Bad — does too many things
def process_user_data(raw_data):
    # validate
    # transform
    # save to database
    # send notification email
    pass

# Good — each function has one job
def validate_user_data(raw_data) -> UserData:
    ...

def save_user(user: UserData) -> None:
    ...

def notify_user(user: UserData) -> None:
    ...
```

### Use Type Hints

Type hints make your code self-documenting and catch bugs before runtime:

```python
def calculate_discount(price: float, percentage: float) -> float:
    """Calculate discounted price."""
    return price * (1 - percentage / 100)
```

## Error Handling

### Be Specific with Exceptions

Catch specific exceptions, not bare `except:`. This prevents accidentally swallowing important errors:

```python
# Bad
try:
    result = api_call()
except:
    pass

# Good
try:
    result = api_call()
except ConnectionError:
    logger.warning("API unavailable, using cached data")
    result = get_cached_result()
except ValueError as e:
    logger.error(f"Invalid response: {e}")
    raise
```

## Virtual Environments

Always use virtual environments to isolate project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

This prevents "it works on my machine" problems and keeps your system Python clean.

## Testing

Write tests alongside your code, not as an afterthought. Use pytest as your test framework:

```python
# test_discount.py
def test_calculate_discount():
    assert calculate_discount(100, 20) == 80.0
    assert calculate_discount(50, 10) == 45.0
    assert calculate_discount(100, 0) == 100.0
```

Run with: `pytest tests/ -v`
