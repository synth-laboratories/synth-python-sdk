import pytest

def test_example():
  assert 1 + 1 == 2

def test_string():
  assert "hello".upper() == "HELLO"

try:
  def test_list():
    assert len([1, 2, 3]) == 3
except Exception as e:
  print(f"Error: {str(e)}")