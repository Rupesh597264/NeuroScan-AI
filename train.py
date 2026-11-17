
import argparse
from pathlib import Path

import cv2
import joblib
import mlflow
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------
# 1. Configuration
# ---------------------------------

IMAGE_SIZE = (64, 64)

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

LABEL_MAP = {
    "NonDemented": 0,
    "MildDemented": 1,
    "ModerateDemented": 1,
    "VeryMildDemented": 1
}

# ---------------------------------
# 2. Load MRI images
# ---------------------------------

def load_dataset(folder):

    images = []
    labels = []
    failures = []

    for class_name in CLASS_NAMES:

        class_folder = folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {class_folder}"
            )

        for image_path in sorted(class_folder.iterdir()):

            if image_path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png", ".bmp"
            ]:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                failures.append(str(image_path))
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

            # Resize to 64 x 64
            resized = cv2.resize(
                gray,
                IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )

            # Normalize pixel values
            normalized = resized.astype(
                np.float32
            ) / 255.0

            # Flatten into 4096 features
            flattened = normalized.flatten()

            images.append(flattened)
            labels.append(LABEL_MAP[class_name])

    if len(images) == 0:
        raise ValueError(f"No readable images in {folder}")

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=int)

    print(f"\nLoaded from: {folder}")
    print("Images:", len(X))
    print("Features:", X.shape[1])
    print("Unreadable files:", len(failures))

    return X, y


# ---------------------------------
# 3. Calculate evaluation metrics
# ---------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        )
    }


# ---------------------------------
# 4. Create classifiers
# ---------------------------------

def create_model(config, random_state):

    if config["algorithm"] == "svm":

        return SVC(
            kernel=config["kernel"],
            C=config["C"],
            class_weight="balanced",
            random_state=random_state
        )

    if config["algorithm"] == "random_forest":

        return RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )

    raise ValueError("Unknown algorithm")


# ---------------------------------
# 5. Main training function
# ---------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train the NeuroScan AI MRI classification model"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"E:\AI\Project\Alzheimer_s Dataset",
        help="Path to the Alzheimer MRI dataset"
    )

    parser.add_argument(
        "--algorithm",
        choices=["all", "svm", "random_forest"],
        default="all"
    ) 

    parser.add_argument(
        "--pca_components",
        type=int,
        default=4
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    parser.add_argument(
        "--svm_c",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--svm_kernels",
        nargs="+",
        choices=["linear", "rbf"],
        default=["rbf", "linear"]
    )

    parser.add_argument(
        "--rf_estimators",
        nargs="+",
        type=int,
        default=[100, 200]
    )
    parser.add_argument(
        "--rf_max_depth",
        type=int,
        default=None,
        help="Maximum depth of Random Forest trees"
    )

    args = parser.parse_args()
    if args.rf_max_depth is not None and args.rf_max_depth <= 0:
        parser.error("--rf_max_depth must be greater than 0")

    dataset_path = Path(args.data_dir)
    if args.pca_components <= 0:
        parser.error("--pca_components must be greater than 0")

    if not 0 < args.test_size < 1:
        parser.error("--test_size must be between 0 and 1")

    if args.svm_c <= 0:
        parser.error("--svm_c must be greater than 0")

    if any(n <= 0 for n in args.rf_estimators):
        parser.error("--rf_estimators must contain positive values")

    RANDOM_STATE = args.random_state
    N_COMPONENTS = args.pca_components
    # Load predefined training and testing folders
    X_train, y_train = load_dataset(
        dataset_path / "train"
    )

    X_test, y_test = load_dataset(
        dataset_path / "test"
    )

    if len(np.unique(y_train)) != 2:
        raise ValueError("Training data must contain both classes.")

    if len(np.unique(y_test)) != 2:
        raise ValueError("Testing data must contain both classes.")

    # Keep the original test folder untouched during model selection
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    print("\nTraining samples:", len(X_fit))
    print("Validation samples:", len(X_val))
    print("Final test samples:", len(X_test))

    # ---------------------------------
    # 6. Fit preprocessing on training only
    # ---------------------------------

    scaler = StandardScaler()

    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)

    pca = PCA(
        n_components=N_COMPONENTS,
        svd_solver="randomized",
        random_state=RANDOM_STATE
    )

    X_fit_pca = pca.fit_transform(X_fit_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    print(
        "\nPCA variance retained:",
        round(
            pca.explained_variance_ratio_.sum() * 100,
            2
        ),
        "%"
    )

    # ---------------------------------
    # 7. Define experiments
    # ---------------------------------

    experiments = []

    # Add SVM experiments
    if args.algorithm in ["all", "svm"]:

        for kernel in args.svm_kernels:

            experiments.append({
                "algorithm": "svm",
                "kernel": kernel,
                "C": args.svm_c
            })

    # Add Random Forest experiments
    if args.algorithm in ["all", "random_forest"]:

        for trees in args.rf_estimators:

            experiments.append({
                "algorithm": "random_forest",
                "n_estimators": trees,
                "max_depth": args.rf_max_depth
            })

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(
        "NeuroScan-Alzheimers-Classification"
    )

    best_score = -1
    best_config = None

    # ---------------------------------
    # 8. Run MLflow experiments
    # ---------------------------------

    print("\nSTARTING EXPERIMENTS")

    for i, config in enumerate(experiments, start=1):

        print(f"\nExperiment {i}: {config}")

        classifier = create_model(config, RANDOM_STATE)

        with mlflow.start_run(
            run_name=f"Experiment_{i}"
        ):

            classifier.fit(X_fit_pca, y_fit)

            predictions = classifier.predict(X_val_pca)

            metrics = calculate_metrics(
                y_val, predictions
            )

            params = {
                key: str(value) if value is None else value
                for key, value in {
                    **config,
                    "pca_components": N_COMPONENTS,
                    "image_size": "64x64",
                    "random_state": RANDOM_STATE
                }.items()
            }

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            print("Validation metrics:", metrics)

            # Select best model using validation F1
            if metrics["f1_score"] > best_score:

                best_score = metrics["f1_score"]
                best_config = config.copy()

    print("\nBEST CONFIGURATION:", best_config)
    print("Best validation F1:", best_score)

    # ---------------------------------
    # 9. Retrain best pipeline
    # ---------------------------------

    print("\nTraining final model on full training data...")

    best_classifier = create_model(best_config, RANDOM_STATE)

    best_pipeline = Pipeline([
        ("scaler", StandardScaler()),

        ("pca", PCA(
            n_components=N_COMPONENTS,
            svd_solver="randomized",
            random_state=RANDOM_STATE
        )),

        ("classifier", best_classifier)
    ])

    best_pipeline.fit(X_train, y_train)

    # Evaluate only after selecting the model
    test_predictions = best_pipeline.predict(X_test)

    test_metrics = calculate_metrics(
        y_test, test_predictions
    )
    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )

    print("\nCONFUSION MATRIX")
    print("Rows = Actual, Columns = Predicted")
    print("Class 0 = NonDemented")
    print("Class 1 = Demented")
    print(cm)

    # Classification Report
    print("\nCLASSIFICATION REPORT")

    print(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["NonDemented", "Demented"],
        zero_division=0
    ))
    print("\nFINAL TEST RESULTS")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # ---------------------------------
    # 10. Save model
    # ---------------------------------

    model_folder = Path("model")
    model_folder.mkdir(exist_ok=True)

    model_path = model_folder / "model.joblib"

    joblib.dump(best_pipeline, model_path)

    print("\nModel saved successfully:", model_path)

    # Log final test evaluation
    with mlflow.start_run(
        run_name="Final_Model_Test_Evaluation"
    ):

        mlflow.log_params({
            key: str(value) if value is None else value
            for key, value in {
                **best_config,
                "pca_components": N_COMPONENTS,
                "selected_using": "validation_f1"
            }.items()
      })

        mlflow.log_metrics({
            f"test_{key}": value
            for key, value in test_metrics.items()
        })

        mlflow.log_artifact(str(model_path))

    print("\nMLflow tracking completed.")
    print("Total experiment runs:", len(experiments) + 1)
    print("Project training completed!")


if __name__ == "__main__":
    main()
