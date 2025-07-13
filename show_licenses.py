import pkg_resources
import prettytable


def get_pkg_license(pkg):
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        lines = pkg.get_metadata_lines('PKG-INFO')
        
    for line in lines:
        if line.startswith('License:'):
            return line[9:]
    return '(Licence not found)'

def print_packages_and_their_licenses():
    """Prints a table of packages and their licenses."""
    table = prettytable.PrettyTable(['Package', 'License'])
    required_packages = {
        "kivy",
        "kivymd",
        "datetime",
        "plyer",
        "random",
        "matplotlib",
        "numpy",
        "tensorflow",
        "sqlite3",
        "rsa",
        "os"
    }
    for package in sorted(pkg_resources.working_set, key=lambda x: str(x).lower()):
        if package.project_name in required_packages:
            table.add_row([str(package), get_pkg_license(package)])
    print(table)
    
if __name__ == "__main__":
    print_packages_and_their_licenses()
