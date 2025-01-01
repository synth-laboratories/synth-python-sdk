import pytest

from synth_sdk.tracing.local import (
    # system_instance_id_var,
    # system_name_var,
    _local,
    system_instance_id_var,
)


@pytest.fixture(autouse=True)
def setup_context():
    # Set context variables
    system_instance_id_var.set("test_system_instance_id")
    # system_instance_id_var.set("test_instance")
    # system_name_var.set("test_agent")

    # Set thread local storage
    _local.system_instance_id = "test_system_instance_id"
    _local.system_name = "test_agent"
    _local.system_instance_id = "test_instance"

    yield
