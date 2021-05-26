import sys
from os.path import join
train_labels = ['apple','banana','beetroot','bell pepper','cabbage','carrot','chilli pepper',
                'corn','cucumber','eggplant','garlic','ginger','grapes','kiwi','lemon','lettuce',
                'mango','onion','orange','paprika','pear','peas','pineapple','pomerganate','potato',
                'spinach','sweetcorn','sweetpotato','tomato','turnip']
train_path = 'train'
valid_path = 'validation'
test_path = 'test'
fixed_size = tuple((200, 200))
epochs = 100
sessions = 1
model_name = 'Model_CNN.h5'
ResNet_path = "fruit_weights_resnet.hdf5"
VGG_path = 'fruit_weights.hdf5'
input_path = 'real_cases'
batch_size = 32   # larger size might not work on some machines