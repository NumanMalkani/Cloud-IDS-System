from model import MODEL_PATH, train_and_save_model


if __name__ == "__main__":
    artifact = train_and_save_model(force=True)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Features: {len(artifact['features'])}")
    print(f"Accuracy: {artifact['accuracy']:.4f}")
    print(f"Confusion matrix: {artifact['confusion_matrix']}")
