if __name__ == '__main__':
    import os
    import sys

    project_path = os.getcwd().split('src')[0]
    sys.path.append(project_path)

    import random
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from src.data.data_format import UtteranceCollection
    from src.model.classifier import SequenceClassifierModel
    from src.data.data_for_training import SequenceClassifierModelDataModule
    from tqdm import tqdm
    from config import config
    from datasets import load_dataset
    from src.data.utils import convert_transformers_dataset_to_utterances, create_balanced_dataset
    from src.model.utils import print_array_stats
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support


    print('############ Preparing data for fine_tuning ################')
    if config.BALANCE == 'True':
        train_dataset = create_balanced_dataset(config.DATASET_NAME, 'train')
        val_dataset = create_balanced_dataset(config.DATASET_NAME, 'validation')
    else:
        train_dataset = load_dataset('silicone', config.DATASET_NAME, split='train')
        val_dataset = load_dataset('silicone', config.DATASET_NAME, split='validation')
    test_dataset = load_dataset('silicone', config.DATASET_NAME, split='test')
    train_utterances = convert_transformers_dataset_to_utterances(train_dataset)
    train_data = UtteranceCollection(utterances=train_utterances)
    print('Training data:', len(train_data.utterances))
    val_utterances = convert_transformers_dataset_to_utterances(val_dataset)
    val_data = UtteranceCollection(utterances=val_utterances)
    print('Validation data:', len(val_data.utterances))
    test_utterances = convert_transformers_dataset_to_utterances(test_dataset)
    test_data = UtteranceCollection(utterances=test_utterances)
    print('Test data:', len(test_data.utterances))


    def train():
        # Prepare the model
        model = SequenceClassifierModel(config=config)

        # Prepare the data
        data = SequenceClassifierModelDataModule(train_data, val_data, test_data, model.tokenizer,
                                                 batch_size=config.BATCH_SIZE)


        # Prepare the trainer
        checkpoint_callback = ModelCheckpoint(dirpath=config.DIR_CHECKPOINTS,
                                              filename=config.FILENAME,
                                              save_top_k=1,
                                              verbose=True,
                                              monitor='val_loss',
                                              mode='min')
        
        logger = TensorBoardLogger(save_dir=config.LOGS.replace('logs', ''),
                                   version=config.VERSION,
                                   name='logs')
        
        trainer = pl.Trainer(callbacks=[checkpoint_callback],
                             max_epochs=config.NUM_EPOCHS,
                             gpus=[0],
                             strategy='dp',
                             logger=logger)

        # Training step
        pl.seed_everything(config.SEED)
        trainer.fit(model, data)


    def verify():
        data = test_data
        random_utterance = random.choice(data.utterances)
        label = random_utterance.label
        ckpt = os.path.join(config.DIR_CHECKPOINTS, f'{config.FILENAME}.ckpt')
        fine_tuned_model = SequenceClassifierModel.load_from_checkpoint(ckpt)
        fine_tuned_model.eval()
        fine_tuned_model.freeze()
        fine_tuned_model.to(fine_tuned_model._device)
        predicted_class = fine_tuned_model.predict_class(random_utterance.text)
        print('Predicted class:', predicted_class)
        print('True class:', label)


    def test():
        data = test_data
        ckpt = os.path.join(config.DIR_CHECKPOINTS, f'{config.FILENAME}.ckpt')
        fine_tuned_model = SequenceClassifierModel.load_from_checkpoint(ckpt)
        fine_tuned_model.eval()
        fine_tuned_model.freeze()
        fine_tuned_model.to(fine_tuned_model._device)

        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1score': []}

        # Compute predictions and evaluate scores
        print("############ Computing predictions #############")
        for utterance in tqdm(data.utterances):
            true = [utterance.label]
            predicted = [fine_tuned_model.predict_class(utterance.text)]
            scores['accuracy'].append(accuracy_score(true, predicted))
            results = precision_recall_fscore_support(true, predicted, average='macro', warn_for=tuple())
            scores['precision'].append(results[0])
            scores['recall'].append(results[1])
            scores['f1score'].append(results[2])

        # Print the results
        print_array_stats(scores['accuracy'], metric='Accuracy', model_name=config.FILENAME, decimal=3)
        print_array_stats(scores['precision'], metric='Precision', model_name=config.FILENAME, decimal=3)
        print_array_stats(scores['recall'], metric='Recall', model_name=config.FILENAME, decimal=3)
        print_array_stats(scores['f1score'], metric='F1 score', model_name=config.FILENAME, decimal=3)
    
    #train()
    verify()
    test()