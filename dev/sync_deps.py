import toml
from pathlib import Path
import re


def read_pyproject_dependencies() -> list:
    """Read dependencies from pyproject.toml."""
    try:
        with open("pyproject.toml", "r") as f:
            config = toml.load(f)
        return config["project"]["dependencies"]
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
        return []


def update_requirements_txt(dependencies: list):
    """Update requirements.txt with dependencies."""
    with open("requirements.txt", "w") as f:
        # Write header comment
        f.write("# Generated from pyproject.toml - do not edit directly\n\n")
        for dep in dependencies:
            f.write(f"{dep}\n")
    print("Updated requirements.txt")


def update_setup_py(dependencies: list):
    """Update setup.py with dependencies."""
    setup_path = Path("setup.py")
    if not setup_path.exists():
        print("setup.py not found - skipping")
        return

    content = setup_path.read_text()

    # Create the new install_requires section
    new_requires = (
        'install_requires=[\n        "'
        + '",\n        "'.join(dependencies)
        + '",\n    ],'
    )

    # Replace the existing install_requires section
    pattern = r"install_requires=\[.*?\],"
    new_content = re.sub(pattern, new_requires, content, flags=re.DOTALL)

    setup_path.write_text(new_content)
    print("Updated setup.py")


def main():
    # Read dependencies from pyproject.toml
    dependencies = read_pyproject_dependencies()
    if not dependencies:
        print("No dependencies found in pyproject.toml")
        return

    # Update both files
    update_requirements_txt(dependencies)
    update_setup_py(dependencies)


if __name__ == "__main__":
    main()
