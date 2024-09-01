from sklearn.pipeline import Pipeline
import torch, sys, pickle
from transformers import BertForSequenceClassification
from pathlib import Path
from movie_review_model.processing.preprocess import TokenizerTransformer, SentimentClassifier
from movie_review_model import model as movie_model
from movie_review_model.config.core import config
from movie_review_model.processing.data_manager import load_custom_dataset, load_imdb_dataset
from movie_review_model.pipeline import build_pipeline

# For model building

def build_model():
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2  # Positive/Negative
    )
    return model

def get_optimizer(model, learning_rate=2e-5):
    # Use PyTorch's AdamW optimizer instead of the deprecated transformers' AdamW
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)

# For defining Pipleine for SentimentClassifier Model

def build_pipeline():
    """
    Build  the pipeline.
    """
    model = movie_model.build_model()
    optimizer = movie_model.get_optimizer(model, 1e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return Pipeline([
        ("tokenizer", TokenizerTransformer()),
        ("classifier", SentimentClassifier(model=model, optimizer=optimizer, device=device)),
    ])

# for prediction pipeline
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from movie_review_model.pipeline import build_pipeline
from movie_review_model.processing.preprocess import TokenizerTransformer
import torch
from transformers import BertForSequenceClassification, BertTokenizer

def predict(text):
    """Predict the sentiment of a text input.

    Args:
        text (str): The text to be classified.

    Returns:
        str: "Positive" or "Negative" based on the prediction.
    """

    # Load the trained model and tokenizer (if not already loaded)
    model = BertForSequenceClassification.from_pretrained(
        "./movie_review_model/trained_models/model"
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare the input data
    input_ids = tokenizer(text, return_tensors="pt", max_length=128).input_ids.to(device)

    # Use the pipeline to predict the sentiment
    with torch.no_grad():
        outputs = model(input_ids)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()

    return {
        "label": predicted_label,
        "sentiment": "Positive" if predicted_label == 1 else "Negative"
    }

if __name__ == "__main__":
    # Example input data
    _review = "The movie was really good and I enjoyed it a lot."
    # Predict sentiment
    _result = predict(_review)
    print(f"Predicted sentiment: {_result}")

# For for model training
def train_pipeline():
    """training pipeline"""
    # Load dataset
    # texts, labels = load_custom_dataset(config.data.train_data_path, split_percentage=25)

    # Load a small sample from the IMDB dataset
    texts, labels = load_imdb_dataset(sample_size=50) #for fast training
    
    # Ensure texts are in the correct format
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("Texts must be a list of strings.")

    # Build pipeline
    pipeline = build_pipeline()

    # Train the model using the pipeline
    pipeline.fit(texts, labels)

    # Save the pipeline (including the trained model)
    # pipeline.named_steps['classifier'].model.save_pretrained(config.output.output_model_path)
    
    #     # Save the pipeline as a .pkl file
    # model_dir = Path(config.output.output_model_path)
    # model_dir.mkdir(parents=True, exist_ok=True)

    # model_path = model_dir / "trained_pipeline.pkl"
    # print(f"Model path: {model_path}")
    # pipeline.named_steps['classifier'].model.save_pretrained(model_path)

    # # save_pipeline(pipeline_to_persist=pipeline)
    # if model_path.exists():
    #     print(f"Model saved successfully at: {model_path}")
    # else:
    #     print("Model save failed.")

    pipeline.named_steps['classifier'].model.save_pretrained("./Movie_Review_Proj/movie_review_model/trained_models/model/trained_pipeline.pkl")
    print("Model and pipeline saved successfully.")

if __name__ == "__main__":
    train_pipeline()
