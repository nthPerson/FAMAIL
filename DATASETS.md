Instructions on how to locate and generate project datasets will be located here in the future.

To train Discriminator model:
1. Generate a dataset using the dataset_generation_tool: 
    - Launch tool dashbaord: `streamlit run discriminator/dataset_generation_tool/app.py`.
    - Choose number of positive and negative pairs in the sidebar.
    - Click "Generate Full Dataset for Download".
    - After dataset is generated, give your dataset a name and click "Save Dataset".
2. Train new model:
    - Launch tool dashboard: `streamlit run discriminator/model/training_dashboard.py`.
    - Select "Train New Model" in the sidebar navigation.
    - Select your dataset in the Dataset Selection $->$ Select Dataset drop-down.
    - Choose hyperparameters (uncheck Bidirectional if ).
    - Give the model a name in the Experiment Name text box.
    - 