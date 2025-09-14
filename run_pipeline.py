import argparse
import os
import sys
import subprocess
from pathlib import Path


def run(cmd, cwd=None):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    ap = argparse.ArgumentParser(description="Run end-to-end ECG transformer forecasting pipeline")
    ap.add_argument('--root_dir', type=str, required=True,
                    help='Root of Rooti data (e.g., D:\\TMU_IIPP\\Nurse_Fatigue_Prof Kang_ Dr Chu\\Rooti)')
    ap.add_argument('--fs', type=int, default=250)
    ap.add_argument('--win_sec', type=int, default=30)
    ap.add_argument('--step_sec', type=int, default=15)
    ap.add_argument('--context_min', type=int, default=60)
    ap.add_argument('--horizon_min', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--out_parquet', type=str, default='data/processed_ecg_segments.parquet')
    ap.add_argument('--max_nurses', type=int, default=4, help='Speed: limit number of nurses for quick run (0 = all)')
    ap.add_argument('--max_zips_per_folder', type=int, default=10, help='Speed: limit zips per nurse (0 = all)')
    ap.add_argument('--max_txt_per_zip', type=int, default=10, help='Speed: limit txt per zip (0 = all)')
    ap.add_argument('--install', action='store_true', help='Install requirements before running')
    args = ap.parse_args()

    # Friendly CPU setting for loky on Windows
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

    project_root = Path(__file__).resolve().parent
    req_file = project_root / 'ecg_transformer_pipeline' / 'requirements.txt'
    data_dir = project_root / 'data'
    outputs_dir = project_root / 'outputs'
    app_out_dir = outputs_dir / 'app'
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    app_out_dir.mkdir(parents=True, exist_ok=True)

    if args.install:
        if req_file.exists():
            run([sys.executable, '-m', 'pip', 'install', '-r', str(req_file)])
        else:
            print(f"Requirements file not found at {req_file}, skipping install.")

    # 1) Preprocessing
    run([
        sys.executable, '-m', 'ecg_transformer_pipeline.preprocessing',
        '--root_dir', args.root_dir,
        '--out_parquet', args.out_parquet,
        '--fs', str(args.fs),
        '--win_sec', str(args.win_sec),
        '--step_sec', str(args.step_sec),
        '--max_nurses', str(args.max_nurses),
        '--max_zips_per_folder', str(args.max_zips_per_folder),
        '--max_txt_per_zip', str(args.max_txt_per_zip),
    ])

    # 2) Train forecaster
    run([
        sys.executable, '-m', 'ecg_transformer_pipeline.train',
        '--data_path', args.out_parquet,
        '--context_min', str(args.context_min),
        '--horizon_min', str(args.horizon_min),
        '--fs', str(args.fs),
        '--win_sec', str(args.win_sec),
        '--step_sec', str(args.step_sec),
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
    ])

    # 3) Application analytics
    predictions_csv = str(outputs_dir / 'test_predictions.csv')
    run([
        sys.executable, '-m', 'ecg_transformer_pipeline.application',
        '--predictions_csv', predictions_csv,
        '--out_dir', str(app_out_dir),
    ])

    # 4) Visualizations
    out_png = str(outputs_dir / 'forecast_vs_actual.png')
    run([
        sys.executable, '-m', 'ecg_transformer_pipeline.visualize',
        '--predictions_csv', predictions_csv,
        '--out_png', out_png,
    ])

    print("\nPipeline completed successfully.")
    print(f"- Segments: {args.out_parquet}")
    print(f"- Best model: outputs/best_transformer.pt")
    print(f"- Predictions (preview): {predictions_csv}")
    print(f"- Application outputs: {app_out_dir / 'application_outputs.csv'}")
    print(f"- Visualization: {out_png}")


if __name__ == '__main__':
    main()



