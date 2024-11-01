from synth_sdk.tracing.decorators import trace_system
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TestClass:
    def __init__(self):
        self.system_id = "test_system"

    @trace_system(event_type="test", manage_event="create", increment_partition=True, verbose=True)
    def test_method(self, x):
        time.sleep(0.1)  # Simulate work
        return x * 2


def test_decorator():
    test = TestClass()
    result = test.test_method(5)
    print(f"Result: {result}")


if __name__ == "__main__":
    test_decorator()
