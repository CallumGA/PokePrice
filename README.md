---
title: PokePrice
emoji: 🔮
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.32.0
app_file: app.py
pinned: false
license: mit
short_description: Predicts Pokémon card price trends with a neural network.
---

## PricePoke: Pokémon Card Price Trend Predictor

This application uses a PyTorch-based neural network to predict whether the market price of a specific Pokémon card will rise by 30% or more over the next six months.

### How It Works
1.  **Select a Card:** Choose a Pokémon card from the dropdown menu. The list is populated from a dataset containing historical price information.
2.  **Get Prediction:** The model analyzes various features of the selected card, such as its rarity, type, and historical price data, to make a prediction.
3.  **View Results:** The application displays:
    *   The prediction (whether the price is expected to **RISE** or **NOT RISE**).
    *   The model's confidence level in the prediction.
    *   A direct link to view the card on TCGPlayer.com.
    *   The actual historical outcome if it exists in the dataset, for comparison.

### The Technology
-   **Model:** A simple feed-forward neural network built with PyTorch.
-   **Data:** The model was trained on a custom dataset derived from the [Pokémon TCG API](https://pokemontcg.io/) and historical market data from TCGPlayer.
-   **Frontend:** The user interface is created with [Gradio](https://www.gradio.app/).
-   **Deployment:** Hosted on [Hugging Face Spaces](https://huggingface.co/spaces).
