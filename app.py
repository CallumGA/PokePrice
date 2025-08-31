import gradio as gr
import torch
import joblib
import pandas as pd
import os
import json
from safetensors.torch import load_file
from typing import List, Tuple
from network import PricePredictor

MODEL_DIR = "model"
DATA_DIR = "data"
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
DATA_PATH = os.path.join(DATA_DIR, "pokemon_final_with_labels.csv")
TARGET_COLUMN = 'price_will_rise_30_in_6m'



def load_model_and_config(model_dir: str) -> Tuple[torch.nn.Module, List[str]]:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        model_config = json.load(f)

    model = PricePredictor(input_size=model_config["input_size"])
    weights_path = os.path.join(model_dir, "model.safetensors")
    model.load_state_dict(load_file(weights_path))
    model.eval()
    return model, model_config["feature_columns"]


def perform_prediction(model: torch.nn.Module, scaler, input_features: pd.Series) -> Tuple[bool, float]:
    features_np = input_features.to_numpy(dtype="float32").reshape(1, -1)
    features_scaled = scaler.transform(features_np)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        logit = model(features_tensor)
        probability = torch.sigmoid(logit).item()
        predicted_class = bool(round(probability))

    return predicted_class, probability

# --- Asset Loading ---
try:
    model, feature_columns = load_model_and_config(MODEL_DIR)
    scaler = joblib.load(SCALER_PATH)
    full_data = pd.read_csv(DATA_PATH)
    ASSETS_LOADED = True
except FileNotFoundError as e:
    print(f"Error loading necessary files: {e}")
    print("Please make sure you have uploaded the 'model' and 'data' directories to your Hugging Face Space.")
    ASSETS_LOADED = False


def predict_price_trend(card_identifier: str) -> str:
    if not ASSETS_LOADED:
        return "## Application Error\nAssets could not be loaded. Please check the logs on Hugging Face Spaces for details. You may need to upload your `model` and `data` directories."
 
    if not card_identifier or not card_identifier.strip().isdigit():
        return "## Input Error\nPlease enter a valid, numeric TCGPlayer ID."

    # --- Find Card Logic ---
    card_id = int(card_identifier.strip())
    card_data = full_data[full_data['tcgplayer_id'] == card_id]

    if card_data.empty:
        return f"## Card Not Found\nCould not find a card with TCGPlayer ID '{card_id}'. Please check the ID and try again."

    # Since tcgplayer_id is unique, we can safely take the first (and only) row.
    card_sample = card_data.iloc[0]
    sample_features = card_sample[feature_columns]

    # --- Prediction Logic ---
    predicted_class, probability = perform_prediction(model, scaler, sample_features)

    prediction_text = "**RISE**" if predicted_class else "**NOT RISE**"
    confidence = probability if predicted_class else 1 - probability
    tcgplayer_id = card_sample['tcgplayer_id']
    tcgplayer_link = f"https://www.tcgplayer.com/product/{tcgplayer_id}?Language=English"

    # --- Output Formatting ---
    true_label_text = ""
    try:
        if TARGET_COLUMN in card_sample and pd.notna(card_sample[TARGET_COLUMN]):
            true_label = bool(card_sample[TARGET_COLUMN])
            true_label_text = f"\n- **Actual Result in Dataset:** The price did **{'RISE' if true_label else 'NOT RISE'}**."
    except (KeyError, TypeError):
        pass # If target column is missing or value is invalid, just skip this part.

    output = f"""
    ## 🔮 Prediction Report for {card_sample['name']}
    - **Prediction:** The model predicts the card's price will {prediction_text} by 30% in the next 6 months.
    - **Confidence:** {confidence:.2%}
    - **View on TCGPlayer:** [Check Current Price]({tcgplayer_link})
    {true_label_text}
    """
    return output


# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="PricePoke Predictor") as demo:
    gr.Markdown(
        """
        # 📈 PricePoke: Pokémon Card Price Trend Predictor
        Enter a Pokémon card's TCGPlayer ID to predict whether its market price will increase by 30% or more over the next 6 months.
        This model was trained on historical TCGPlayer market data.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            card_input = gr.Textbox(
                label="TCGPlayer ID",
                placeholder="e.g., '84198'",
                info="Find the ID in the card's URL on TCGPlayer's website (e.g., tcgplayer.com/product/84198/... has ID 84198)."
            )
            predict_button = gr.Button("Predict Trend", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### Example Cards")
            if ASSETS_LOADED:
                example_df = full_data.sample(5, random_state=42)[['name', 'tcgplayer_id']]
                gr.Markdown(example_df.to_markdown(index=False))
            else:
                gr.Markdown("Could not load examples.")

        with gr.Column(scale=2):
            output_markdown = gr.Markdown()

    predict_button.click(fn=predict_price_trend, inputs=[card_input], outputs=[output_markdown])
    card_input.submit(fn=predict_price_trend, inputs=[card_input], outputs=[output_markdown])

if __name__ == "__main__":
    demo.launch()