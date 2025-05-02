# Mixture_of_Experts

We're building a Mixture of Experts (MoE) model with three expert networks, each trained on a different subset of the data. A gating network will dynamically select the top two experts to contribute to the final prediction.

This approach allows the model to specialize across different data distributions while still combining diverse perspectives for better accuracy.
