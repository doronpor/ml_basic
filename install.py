import os
import subprocess
import sys


def install_requirements() -> None:
    """Install both main and development requirements."""
    print("Installing main requirements...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )

    print("\nInstalling development requirements...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"]
    )

    print("\nSetting up pre-commit hooks...")
    subprocess.check_call([sys.executable, "-m", "pre_commit", "install"])

    print("\nAll dependencies installed successfully!")


if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    try:
        install_requirements()
    except subprocess.CalledProcessError as e:
        print(f"\nError during installation: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)
