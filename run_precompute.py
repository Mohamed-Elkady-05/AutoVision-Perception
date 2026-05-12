from video_sequence_preprocessing import SequenceFeaturePrecomputer

if __name__ == '__main__':
    precomputer = SequenceFeaturePrecomputer(sequence_dir='./data/preprocessed_sequences', output_dir='./cache/vgg16_sequence_features')
    precomputer.precompute_sequences(split='train')
