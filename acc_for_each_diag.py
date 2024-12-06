import os
import shutil
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from itertools import cycle

# GPU settings
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
from train import init_way
from prepare_data import load_and_preprocess_image
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from statsmodels.stats.contingency_tables import mcnemar,cochrans_q

def plot_roc(labels, predictions, init_way_, NUM_CLASSES=3, **kwargs):
    if '_' in init_way_:
        init_way_ = init_way_.replace('_', " ")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green', 'deeppink', 'darkorchid', 'coral', 'slateblue', 'gold','orange'])
    plt.figure(figsize=(5, 5))
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(id_cls[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.09)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve {init_way_}')
    plt.legend(loc="lower right")
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(f"ROCplot/{init_way_}.png")
    return roc_auc

def get_class_id(image_root):
    id_cls = {}
    for i, item in enumerate(os.listdir(image_root)):
        if os.path.isdir(os.path.join(image_root, item)):
            id_cls[i] = item
    return id_cls

def get_flops(model, model_inputs) -> float:
        """
        Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
        in inference mode. It uses tf.compat.v1.profiler under the hood.
        """
        if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
        ):
            raise ValueError(
                "Calculating FLOPS is only supported for "
                "`tf.keras.Model` and `tf.keras.Sequential` instances."
            )
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )
 
        # Compute FLOPs for one sample
        batch_size = 1
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
            for inp in model_inputs
        ]
 
        # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
        real_model = tf.function(model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)
 
        # Calculate FLOPs with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
            )
            .with_empty_output()
            .build()
        )
 
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )
 
        tf.compat.v1.reset_default_graph()
 
        # convert to GFLOPs
        return (flops.total_float_ops / 1e9)/2  #tf.compat.v1.profiler.profile 默认会统计模型完整的 FLOPs，包括前向传播和反向传播的运算。因此，为了只统计推理时的 FLOPs，代码将总 FLOPs 除以 2。


def stats_model(model):
    flops = tf.compat.v1.profiler.profile(model, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(model, options=tf.compat.v1.profiler.ProfileOptionBuilder.total_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
    return flops.total_float_ops,  params.total_parameters

if __name__ == '__main__':
    IMAGE_WIDTH,IMAGE_HEIGHT,CHANNELS = 128,128,3
    NUM_class = 3
    index_col = []
    index_start = 0
    our_model_loc = 0
    test_name = 'benchmark' # benchmark or pretain
    if test_name == 'benchmark':
        jobs = [
                'JaunENet',
                'Xception',
                'VGG',
                'Vit',
                'Mobilenet',
                'Inception',
                'Resnet',
                'Densenet',
                'ConvNeXt'
                ]
    elif test_name == 'pretain':
        jobs = [
                'EDID_weakly_labeled',
                'EDID',
                'ISIC_weakly_labeled',
                'ISIC',
                'Imagenet',
                'JaunENet',
                ]
    else:
        raise ValueError("no such option !")
    res_dict = dict()
    for job in jobs:
        if job == 'Imagenet':
            IMAGE_WIDTH,IMAGE_HEIGHT = 224, 224
        else:
            IMAGE_WIDTH,IMAGE_HEIGHT = 128, 128
        init_way_ = job
        model = init_way(init_way_,IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, NUM_class, show_summary=False)
        save_model_dir = "saved_model_{}/".format(init_way_)
        weight_file_name = save_model_dir + '/best_valid_acc_model_weights.h5'
        model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        model.load_weights(filepath=weight_file_name)

        train_dir = './dataset/test/'
        filenames = os.listdir(train_dir)
        res = {}
        preds = []
        preds_or = []
        labels = []
        time_cost_list = []
        for diag_name in filenames:
            correct = {}
            print(diag_name)
            images_dir = os.listdir(train_dir + diag_name + '/')
            diag_images_num = len(images_dir)
            for image_name in images_dir:
                image_raw = tf.io.read_file(train_dir + '/' + diag_name + '/' + image_name)
                image_tensor = load_and_preprocess_image(image_raw,IMAGE_WIDTH=IMAGE_WIDTH,IMAGE_HEIGHT=IMAGE_HEIGHT, data_augmentation=False)
                image_tensor = tf.expand_dims(image_tensor, axis=0)
                start_time = time.time()
                pred = model(image_tensor, training=False)
                end_time = time.time()
                time_cost_list.append((end_time-start_time))
                preds_or.append(pred.numpy()[0])
                idx = tf.math.argmax(pred, axis=-1).numpy()[0]
                preds.append(idx)
                id_cls = get_class_id("./original_dataset")
                labels.append(list(id_cls.values()).index(diag_name))
                img_class = id_cls[idx]
                if diag_name != img_class:
                    error_dir = './error_images/' + diag_name + '/' + img_class + '/'
                    if not os.path.exists(error_dir):
                        os.makedirs(error_dir)
                    shutil.copyfile(train_dir + '/' + diag_name + '/' + image_name, error_dir + image_name)
                if img_class not in correct.keys():
                    correct[img_class] = 1
                else:
                    correct[img_class] += 1
            #     print("\r The predicted category of this picture is: {}".format(img_class), end="", flush=True)
            # print("\nthe accuracy of {} is {:.2f}".format(diag_name, correct[diag_name] / diag_images_num), correct)
            res[diag_name + "_T"] = correct
        ress = pd.DataFrame(res)
        ress.sort_index(ascending=True, axis=0, inplace=True)
        ress = ress.fillna(0)
        ress.to_csv("./log/acc_for_diag" + weight_file_name[-6:-3] + ".csv")
        report_dict = classification_report(labels, preds,  output_dict=True)
        sens_list = []
        spec_list = []
        for _ in range(NUM_class):
            slabels = [1 if i==_  else 0 for i in labels]
            spreds = [1 if i==_  else 0 for i in preds]
            tn, fp, fn, tp = confusion_matrix(slabels, spreds).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            sens_list.append(sensitivity)
            spec_list.append(specificity)
        # print(classification_report(labels, preds, digits=6))
        labels_or = tf.one_hot(labels, depth=3).numpy()
        if test_name == 'benchmark':
            tname = 'of '+init_way_
        else:
            tname = 'pretrained on '+init_way_
        roc_auc = plot_roc(labels=np.array(labels_or), predictions=np.array(preds_or),init_way_=tname , NUM_class=NUM_class)

        # flops,params = stats_model(model)
        
        # classsify_res['params'] = report_dict['params']
        # print(flops,params)
        classsify_res = dict()
        classsify_res['params(M)']=model.count_params()/1e6
        classsify_res['GFLOPs'] =get_flops(model, [image_tensor])
        classsify_res['acc'] = report_dict['accuracy']
        classsify_res['macro-f1'] = report_dict['macro avg']['f1-score']
        classsify_res['macro-recall'] = report_dict['macro avg']['recall']
        classsify_res['macro-precision'] = report_dict['macro avg']['precision']
        classsify_res['macro-sensitivity'] = np.mean(sens_list)
        classsify_res['macro-specificity'] = np.mean(specificity)
        classsify_res['macro-auc'] = np.mean(list(roc_auc.values()))
        classsify_res['Time cost'] = np.mean(time_cost_list)
        index_start+=1
        if job == jobs[our_model_loc]:
            model_A_preds = np.array(preds)
            true_labels = np.array(labels)
            classsify_res['p_value'] = None
        else:
            model_B_preds = np.array(preds)
            n_01 = np.sum((model_A_preds != true_labels) & (model_B_preds == true_labels))  # Model A incorrect, Model B correct
            n_10 = np.sum((model_A_preds == true_labels) & (model_B_preds != true_labels))  # Model A correct, Model B incorrect
            n_00 = np.sum((model_A_preds != true_labels) & (model_B_preds != true_labels))  # Both incorrect
            n_11 = np.sum((model_A_preds == true_labels) & (model_B_preds == true_labels))  # Both correct

            # Create the 2x2 contingency table
            contingency_table = [[n_11, n_01],  # First row: Model A and B correct, Model B only correct
                                [n_10, n_00]]  # Second row: Model A only correct, both incorrect

            # Perform McNemar's test
            result = mcnemar(contingency_table, exact=False,correction=False)
            classsify_res['p_value'] = result.pvalue

        res_dict[init_way_] = classsify_res
        pd.DataFrame(res_dict).T.to_csv(f"{test_name}_Test_result.csv")
        print(pd.DataFrame(res_dict).T)