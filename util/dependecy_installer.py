from importlib import import_module
from subprocess import check_call


class DependencyInstaller:
    GREEN = "\033[1m\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m\033[1m"

    @staticmethod
    def install_dependencies(packages_list: list) -> None:
        """
        Check for the presence of packages in the given list and install them if not found.

        Args:
        - packages_list (list): List of packages to check and install if not found

        Returns:
        - None
        """
        for package in packages_list:
            try:
                import_module(package)  # Try to import the package
            except ImportError:
                # If the package is not found, attempt to install it
                print(f"{DependencyInstaller.BLUE}{package}{DependencyInstaller.RESET} is not installed. Installing...")
                try:
                    check_call(["pip", "install", package])
                    print(
                        f"{DependencyInstaller.BLUE}{package}{DependencyInstaller.GREEN} has been successfully installed{DependencyInstaller.RESET}")
                except Exception as e:
                    print(f"{DependencyInstaller.RED}Error installing {package}: {str(e)}{DependencyInstaller.RESET}")
