import argparse
import numpy as np
import os.path as osp

framework_root = osp.dirname(osp.realpath(__file__))
data_experiments_address_csv = osp.join(framework_root, 'Data/HR_GSR_Data.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str, default='FC',
                        help='model type to train: (FC, ...)')
    io_args = parser.parse_args()
    modelName = io_args.model

    np.random.seed(42)  # to make model making reproducible for comparisons

    if modelName == "FC":
        # network_name = 'Simple_FC_LR0001_Period10_Total5_DsGames_Relu_Net'
        # external_dict = {'model_name': network_name}
        # from Keras_Nets.Simple_Fully_Connected import SimpleFcModel
        # model = SimpleFcModel(external_dict)

        # network_name = 'Simple_FC_Softmax_LR0001_decay10Ep_Period10_Total5_DsGames_GSR_Net'
        # external_dict = {'model_name': network_name}
        # from Keras_Nets.Simple_Fully_Connected import SimpleFcModel_Softmax
        # model = SimpleFcModel_Softmax(external_dict)

        network_name = 'Simple_FC_HR_GSR_DEEP_Softmax_LR0001_decay10Ep_Period10_Total5_DsGames_Net'
        external_dict = {'model_name': network_name}
        from Keras_Nets.Simple_Fully_Connected import SimpleFcModel_Softmax_HR_GSR
        model = SimpleFcModel_Softmax_HR_GSR(external_dict)

        model.set_data(data_address_csv=data_experiments_address_csv)
        model.model_arch()
        model.set_solver()
        model.train_validate_Optimized()

    elif modelName == "CNN":
        # TODO: CODE FOR CALLING CNN CLASS GOES IN HERE
        raise Exception("Code for model {} not implemented" .format(modelName))
    elif modelName == "lin-reg":
        # TODO: CODE FOR CALLING LINEAR REGRESSION CLASS GOES IN HERE
        raise Exception("Code for model {} not implemented" .format(modelName))
    else:
        raise Exception("Unknown model: {}" .format(modelName))