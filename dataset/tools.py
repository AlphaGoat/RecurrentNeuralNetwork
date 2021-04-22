import csv
import os
import re
import shutil

import random

def read_data(filename):

    instance = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(filename, delimiter=',')

        for row in csv_reader:
            # each row corresponds to one time interval
            data = [float(val) for val in row[:10]]
            instance.append(data)

    return instance

def scroll_directory(dirpath):
    """
    scroll directory for datafiles and return all data filepaths
    found as array
    """

    for root, dirs, files in os.walk(".", topdown=False):

        for name in files:
            if name.endswith('.sign'):
                filepath = os.path.join(root, name)

def reformat_vowel_trainset(datafile,
                            datadir):
    """
    Reorganize the japanese vowel data into format
    that my current c++ utility functions can read

    :params:
        :datafile: path to japanese vowel datafile 
        :datadir:  directory where we want to save data
    """
    if os.path.exists(datadir): shutil.rmtree(datadir)

    os.mkdir(datadir)

    NUM_BLOCKS = 30 # number of blocks per speaker

    person_id = 0
    block_iterator = 0
    with open(datafile, 'r') as f:

        csv_file = csv.reader(f)

        # iterate over lines until we reach the next
        # blank line. When that occurs, save into new 
        # file
        data_block = []
        for i, line in enumerate(csv_file):

            if not line:
                # write out datafile
                outdir = os.path.join(datadir, "person{}".format(person_id))

                try:
                    os.mkdir(outdir)
                except FileExistsError:
                    pass

                filename = "block{}.csv".format(block_iterator)
                outfile = os.path.join(outdir, filename)


                # write out current blocks
                with open(outfile, "w") as out:
                    csv_writer = csv.writer(out)
                    for l in data_block:
                        csv_writer.writerow(l)

                # clear data block 
                data_block.clear()

                block_iterator+=1
                if block_iterator % 30 == 0:
                    block_iterator = 0
                    person_id += 1

            else:
                data_block.append(line)

def reformat_vowel_testset(datafile,
                            datadir):
    """
    Same as with training data set, but the data 
    frequency for each speaker is different
    """
    freqs = [31, 35, 88, 44, 29, 24, 40, 50, 29]

    # pointer to let us know which person we are on
    pointer = 0

    if os.path.exists(datadir): shutil.rmtree(datadir)

    os.mkdir(datadir)
    
    with open(datafile, 'r') as f:

        csv_file = csv.reader(f)

        data_block = []
        num_iter = 0
        for line in csv_file:

            if not line:
                # write out datafile
                outdir = os.path.join(datadir, "person{}".format(pointer))

                try:
                    os.mkdir(outdir)
                except FileExistsError:
                    pass

                filename = "block{}.csv".format(num_iter)
                outfile = os.path.join(outdir, filename)


                # write out current blocks
                with open(outfile, "w") as out:
                    csv_writer = csv.writer(out)
                    for l in data_block:
                        csv_writer.writerow(l)

                # clear data block 
                data_block.clear()

                num_iter+=1
                if num_iter == freqs[pointer]:
                    num_iter = 0
                    pointer += 1

            else:
                data_block.append(line)

def relabel_valid_set(trainpath,
                      validpath):
    """
    Since there are actually more valid instances than 
    training instances, I'm going to move over some of the 
    validation instances to training. To do that, I'm going 
    to need to rename the valid instances so they don't overwrite
    any of the training instances 
    """
    highest_label_dict = {}
    for subdir, dirs, files in os.walk(trainpath):
        for d in dirs:
            # Return highest label number
            highest_index = None
            subdir_name = os.path.join(trainpath, d)
            for file in os.scandir(subdir_name):
                index_num = int(re.findall("\d+", file.name)[0])
                if not highest_index or index_num > highest_index:
                    highest_index = index_num

            highest_label_dict[d] = highest_index

    # Now walk through validation directory and rename data instances
    for subdir, dirs, files in os.walk(validpath):
        for d in dirs:
            subdir_name = os.path.join(validpath, d)
            for file in os.scandir(subdir_name):
                filename = file.name
                index_num = int(re.findall("\d+", filename)[0])
                new_index = index_num + highest_label_dict[d] + 1
                newname = filename.replace(str(index_num), str(new_index))

                oldpath = os.path.join(subdir_name, filename)
                newpath = os.path.join(subdir_name, newname)

                # rename valid file with new index
                os.rename(oldpath, newpath)

def encode_iris_classifications():
    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    file = "iris/iris-test.txt"

    with open(file, 'r') as f:

        csv_file = csv.reader(f)

        lines = []
        for l in csv_file:

            l = l[0].split(' ')
            line = [float(val) for val in l[:-1]]
            line.append(float(classes.index(l[-1])))

            lines.append(line)

        outfile = "iris/encoded-iris-test.txt"
        with open(outfile, "w") as out:
            csv_writer = csv.writer(out)
            for l in lines:
                csv_writer.writerow(l)

def get_whole_batch(trainpath,
                    validpath):
    """
    Get all training instances for one person and divide into training,
    validation, and testing based on random 80-20-20 split 
    """
    # Directory names for new train and valid paths 
    new_traindir = "dataset/vowels/train/"
    if os.path.exists(new_traindir): shutil.rmtree(new_traindir)
    os.mkdir(new_traindir)

    new_validdir = "dataset/vowels/val/"
    if os.path.exists(new_validdir): shutil.rmtree(new_validdir)
    os.mkdir(new_validdir)

    new_testdir = "dataset/vowels/test/"
    if os.path.exists(new_testdir): shutil.rmtree(new_testdir)
    os.mkdir(new_testdir)

    # dict mapping person to list of instances
    instance_list = {}

    # map instances for each person to their path
    instance_to_path = {}

    traindirs = os.listdir(trainpath)
    for d in traindirs:

        # keep a dictionary of instances for each person in dataset
        instance_list[d] = []
        person_instances = {}
        subdir_path = os.path.join(trainpath, d)
        for file in os.scandir(subdir_path):

            filename = file.name
            instance_list[d].append(filename)

            filepath = os.path.join(subdir_path, filename)
            person_instances[filename] = filepath

        instance_to_path[d] = person_instances
        
    validdirs = os.listdir(validpath)
    for d in validdirs:

        # keep a dictionary of instances for each person in dataset
        subdir_path = os.path.join(validpath, d)
        for file in os.scandir(subdir_path):

            filename = file.name
            instance_list[d].append(filename)

            filepath = os.path.join(subdir_path, filename)
            instance_to_path[d][filename] = filepath

    # Randomly shuffle data instances for each person 
    for person_id in instance_list:
        paths_for_person_id = instance_to_path[person_id]
        instances = instance_list[person_id]

        train_split = int(.8 * len(instances))
        val_split = int(.1 * len(instances))
        test_split = int(.1 * len(instances))

        # randomly sample 80% for train, 10% for val, 10% for test
        train_samples = random.sample(instances, k=train_split)
        for sample in train_samples: instances.remove(sample)

        val_samples = random.sample(instances, k=val_split)
        for sample in val_samples: instances.remove(sample)

        test_samples = random.sample(instances, k=test_split)
        for sample in test_samples: instances.remove(sample)

        # Add new samples to train set 
        new_trainpath = os.path.join(new_traindir, person_id)
        if not os.path.isdir(new_trainpath):
            os.mkdir(new_trainpath)

        # if there are any files in the directory, remove them
        for root, dirs, files in os.walk(new_trainpath):
            for file in files:
                os.remove(os.path.join(root, file))

        for sample in train_samples:
            oldpath = paths_for_person_id[sample]
            newpath = os.path.join(new_trainpath, sample)
            shutil.copy(oldpath, newpath)

        # Add new samples to valid set 
        new_valpath = os.path.join(new_validdir, person_id)
        if not os.path.isdir(new_valpath):
            os.mkdir(new_valpath)
        for root, dirs, files in os.walk(new_valpath):
            for file in files:
                os.remove(os.path.join(root, file))

        for sample in val_samples:
            oldpath = paths_for_person_id[sample]
            newpath = os.path.join(new_valpath, sample)
            shutil.copy(oldpath, newpath)

        # add new samples to test set 
        new_testpath = os.path.join(new_testdir, person_id)
        if not os.path.isdir(new_testpath):
            os.mkdir(new_testpath)
        for root, dirs, files in os.walk(new_testpath):
            for file in files:
                os.remove(os.path.join(root, file))

        for sample in test_samples:
            oldpath = paths_for_person_id[sample]
            newpath = os.path.join(new_testpath, sample)
            shutil.copy(oldpath, newpath)

def reformat_sign_dataset():
    """
    Reorganize sign dataset into training, validation,
    and test sets 
    """
    parentdir = "dataset"
    signdir = os.path.join(parentdir, 'signs')
    split = [.8, .1, .1] #train/val/test split
    data_dict = {}

    # Create train, validation, and test folders with directories 
    # for all labels 
    trainpath = os.path.join(signdir, "train")
    validpath = os.path.join(signdir, "valid")
    testpath = os.path.join(signdir, "test")
    if os.path.exists(trainpath): shutil.rmtree(trainpath)
    if os.path.exists(validpath): shutil.rmtree(validpath)
    if os.path.exists(testpath): shutil.rmtree(testpath)

    for root, dirs, files in os.walk(signdir):
        # Get directory names (labels for dataset)
        if files:
            # Get name of root ("label")
            label = ''.join([i for i in root if not i.isdigit()])
            label = label.split('/')[-1]

            if label not in data_dict: data_dict[label] = []
            for f in files:
                data_dict[label].append(os.path.join(root, f))

    os.mkdir(trainpath)
    for key in data_dict:
        folderpath = os.path.join(trainpath, key)
        os.mkdir(folderpath)

    os.mkdir(validpath)
    for key in data_dict:
        folderpath = os.path.join(validpath, key)
        os.mkdir(folderpath)

    os.mkdir(testpath)
    for key in data_dict:
        folderpath = os.path.join(testpath, key)
        os.mkdir(folderpath)

    # Split data into train, validation and test sets 
    for key in data_dict:
        files = data_dict[key]
        num_train = int(len(files) * split[0])
        num_val = int(len(files) * split[1])
        num_test = len(files) - num_train - num_val

        trainsamples = random.sample(files, num_train)
        for sample in trainsamples: files.remove(sample)

        validsamples = random.sample(files, num_val) 
        for sample in validsamples: files.remove(sample)

        testsamples = random.sample(files, num_test)
        for sample in testsamples: files.remove(sample)

        for sample in trainsamples:
            filename = sample.split('/')[-1]
            newpath = os.path.join(trainpath, key, filename)
            shutil.copy(sample, newpath)
            # remove sample after copying
            os.remove(sample)

        for sample in validsamples:
            filename = sample.split('/')[-1]
            newpath = os.path.join(validpath, key, filename)
            shutil.copy(sample, newpath)
            # remove sample after copying 
            os.remove(sample)

        for sample in testsamples:
            filename = sample.split('/')[-1]
            newpath = os.path.join(testpath, key, filename)
            shutil.copy(sample, newpath)
            # remove sample after copying 
            os.remove(sample)

    # Remove original folders
    for root, dirs, files in os.walk(signdir):
        if dirs:
            for d in dirs:
                if d != "train" and d != "test" and d != "valid":
#                    shutil.rmtree(signdir, d)
                    fullpath = os.path.join(signdir, d)
                    if os.path.exists(fullpath): shutil.rmtree(fullpath)

if __name__ == '__main__':

    datadir = "dataset"
    vowel_datadir = os.path.join(datadir, "vowels")

    if os.path.exists(vowel_datadir): shutil.rmtree(vowel_datadir)
    os.mkdir(vowel_datadir)

    train_datadir = os.path.join(vowel_datadir, "temp_train")
    if os.path.exists(train_datadir): shutil.rmtree(train_datadir)
    os.mkdir(train_datadir)

    train_datafile = os.path.join(datadir, "ae.train")
    reformat_vowel_trainset(train_datafile, train_datadir)
#
    val_datadir = os.path.join(vowel_datadir, "temp_val")
    if os.path.exists(val_datadir): shutil.rmtree(val_datadir)
    os.mkdir(val_datadir)

    val_datafile = os.path.join(datadir, "ae.test")
    reformat_vowel_testset(val_datafile, val_datadir)
#
    relabel_valid_set(train_datadir, val_datadir)
    get_whole_batch(train_datadir, val_datadir)
    
    # remove temp directories 
    shutil.rmtree(val_datadir)
    shutil.rmtree(train_datadir)
#
#    encode_iris_classifications()

    reformat_sign_dataset()
