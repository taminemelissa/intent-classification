if __name__ == '__main__':
    import os
    import sys
    project_dir = os.getcwd().split('src')[0]
    sys.path.append(project_dir)

    from datasets import load_dataset
    from src.data.utils import convert_transformers_dataset_to_passages

    datasets_names = ['dyda_da', 'dyda_e', 'iemocap', 'maptask', 'meld_e',   
                      'meld_s', 'mrda', 'oasis', 'sem', 'swda']

    for name in datasets_names:
        train_dataset = load_dataset('silicone', name, split='train')
        val_dataset = load_dataset('silicone', name, split='validation')
        test_dataset = load_dataset('silicone', name, split='test')

