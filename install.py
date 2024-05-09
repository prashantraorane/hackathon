import subprocess
import sys

def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} successfully installed.")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}.")

if __name__ == "__main__":
    package_name = "llama_index.core"  # replace with the actual name of the package if different
    install_package(package_name)
