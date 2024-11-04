import ast
import synth_sdk.config.settings
import os
from collections import defaultdict


def get_imports(file_path):
    """Extract all imports from a Python file."""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append(name.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            for name in node.names:
                full_name = f"{module}.{name.name}" if module else name.name
                imports.append(full_name)
    return imports


def find_circular_imports(directory):
    """Find potential circular imports in a directory."""
    # Map files to their imports
    file_imports = {}
    # Map modules to the files that import them
    imported_by = defaultdict(list)

    # Walk through all Python files
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                module_path = os.path.relpath(file_path, directory).replace("/", ".")[
                    :-3
                ]

                try:
                    imports = get_imports(file_path)
                    file_imports[module_path] = imports

                    # Record which files import this module
                    for imp in imports:
                        imported_by[imp].append(module_path)
                except SyntaxError:
                    print(f"Syntax error in {file_path}")
                    continue

    # Check for circular dependencies
    def check_circular(module, visited, path):
        if module in path:
            cycle = path[path.index(module) :] + [module]
            return f"Circular import detected: {' -> '.join(cycle)}"

        if module in visited:
            return None

        visited.add(module)
        path.append(module)

        if module in file_imports:
            for imp in file_imports[module]:
                result = check_circular(imp, visited.copy(), path.copy())
                if result:
                    return result

        return None

    # Check each module
    for module in file_imports:
        result = check_circular(module, set(), [])
        if result:
            print(result)


if __name__ == "__main__":
    # Use the current directory, or specify your package directory
    directory = "synth_sdk"
    find_circular_imports(directory)
