from pathlib import Path
import sys
import os

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from utils.preprocessing import load_data, preprocess_data


class PropertyDataCleaner:
    def __init__(self, input_dir: str, output_dir: str, artifacts_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.artifacts_dir = Path(artifacts_dir)

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def clean(self):
        # Load data
        house_file = self.input_dir / "house_main.csv"
        apartment_file = self.input_dir / "apartment_main.csv"

        # Process
        df = load_data(str(house_file), str(apartment_file))
        cleaned_df = preprocess_data(df, self.artifacts_dir)

        # Save
        output_file = self.output_dir / "properties_clean.csv"
        cleaned_df.to_csv(output_file, index=False)
        return {
            "file_path": str(output_file),
            "records": len(cleaned_df),
            "artifacts_dir": str(self.artifacts_dir),
        }


if __name__ == "__main__":
    # When running in container, use environment variables or default paths
    input_dir = os.getenv("INPUT_DIR", "/app/data/raw")
    output_dir = os.getenv("OUTPUT_DIR", "/app/data/processed")
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "/app/data/artifacts")

    print(f"Starting cleaner with paths:")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Artifacts dir: {artifacts_dir}")

    cleaner = PropertyDataCleaner(
        input_dir=input_dir, output_dir=output_dir, artifacts_dir=artifacts_dir
    )
    cleaner.clean()
