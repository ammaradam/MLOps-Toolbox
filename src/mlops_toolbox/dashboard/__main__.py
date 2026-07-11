"""Launch the dashboard: `python -m mlops_toolbox.dashboard`."""

import argparse

from mlops_toolbox.dashboard.app import run_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mlops-toolbox dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument(
        "--registry-path",
        default=None,
        help="Path to the project registry JSON file (defaults to ~/.mlops-toolbox/projects.json, "
        "or $MLOPS_TOOLBOX_HOME/projects.json if that env var is set).",
    )
    args = parser.parse_args()
    run_dashboard(host=args.host, port=args.port, registry_path=args.registry_path)


if __name__ == "__main__":
    main()
