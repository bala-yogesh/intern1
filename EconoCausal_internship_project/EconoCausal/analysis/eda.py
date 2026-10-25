"""Create a basic exploratory analysis and save a chart."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt

# Permit running this file directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import generate_synthetic_data  # noqa: E402
from src.metrics import summarize_data  # noqa: E402


def main() -> None:
    data = generate_synthetic_data()
    print("Descriptive summary")
    for name, value in summarize_data(data).items():
        print(f"{name}: {value:.3f}")

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data["price"], data["units_sold"], alpha=0.35, s=12)
    ax.set(title="Synthetic price vs. units sold", xlabel="Price", ylabel="Units sold")
    fig.tight_layout()
    output_path = output_dir / "price_vs_demand.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
