************************
MatchZoo Model Reference
************************

DenseBaseline
#############

Model Documentation
*******************

A simple densely connected baseline model.

Examples:
    >>> model = DenseBaseline()
    >>> model.params['mlp_num_layers'] = 2
    >>> model.params['mlp_num_units'] = 300
    >>> model.params['mlp_num_fan_out'] = 128
    >>> model.params['mlp_activation_func'] = 'relu'
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================================================  ======================================================  ======================================================================
  ..  Name                         Description                                                                                                                Default Value                                           Default Hyper-Space
====  ===========================  =========================================================================================================================  ======================================================  ======================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.dense_baseline.DenseBaseline'>
   1  task                         Decides model output shape, loss, and metrics.
   2  out_activation_func          Activation function used in output layer.
   3  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding                    FloatTensor containing weights for the Embedding.
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  padding_idx                  If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.                                               True
  10  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                                                          256                                                     quantitative uniform distribution in  [16, 512), with a step size of 1
  11  mlp_num_layers               Number of layers of the multiple layer percetron.                                                                          3                                                       quantitative uniform distribution in  [1, 5), with a step size of 1
  12  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.                                    64                                                      quantitative uniform distribution in  [4, 128), with a step size of 4
  13  mlp_activation_func          Activation function used in the multiple layer perceptron.                                                                 relu
====  ===========================  =========================================================================================================================  ======================================================  ======================================================================

DSSM
####

Model Documentation
*******************

Deep structured semantic model.

Examples:
    >>> model = DSSM()
    >>> model.params['mlp_num_layers'] = 3
    >>> model.params['mlp_num_units'] = 300
    >>> model.params['mlp_num_fan_out'] = 128
    >>> model.params['mlp_activation_func'] = 'relu'
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================  ===================================  =====================================================================
  ..  Name                         Description                                                                                Default Value                        Default Hyper-Space
====  ===========================  =========================================================================================  ===================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.dssm.DSSM'>
   1  task                         Decides model output shape, loss, and metrics.
   2  out_activation_func          Activation function used in output layer.
   3  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   4  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
   5  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
   6  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                   quantitative uniform distribution in  [4, 128), with a step size of 4
   7  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
   8  vocab_size                   Size of vocabulary.                                                                        419
====  ===========================  =========================================================================================  ===================================  =====================================================================

CDSSM
#####

Model Documentation
*******************

CDSSM Model implementation.

Learning Semantic Representations Using Convolutional Neural Networks
for Web Search. (2014a)
A Latent Semantic Model with Convolutional-Pooling Structure for
Information Retrieval. (2014b)

Examples:
    >>> import matchzoo as mz
    >>> model = CDSSM()
    >>> model.params['task'] = mz.tasks.Ranking()
    >>> model.params['vocab_size'] = 4
    >>> model.params['filters'] =  32
    >>> model.params['kernel_size'] = 3
    >>> model.params['conv_activation_func'] = 'relu'
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================  =====================================  =====================================================================
  ..  Name                         Description                                                                                Default Value                          Default Hyper-Space
====  ===========================  =========================================================================================  =====================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.cdssm.CDSSM'>
   1  task                         Decides model output shape, loss, and metrics.
   2  out_activation_func          Activation function used in output layer.
   3  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   4  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                    quantitative uniform distribution in  [8, 256), with a step size of 8
   5  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                      quantitative uniform distribution in  [1, 6), with a step size of 1
   6  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                     quantitative uniform distribution in  [4, 128), with a step size of 4
   7  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
   8  vocab_size                   Size of vocabulary.                                                                        419
   9  filters                      Number of filters in the 1D convolution layer.                                             3
  10  kernel_size                  Number of kernel size in the 1D convolution layer.                                         3
  11  conv_activation_func         Activation function in the convolution layer.                                              relu
  12  dropout_rate                 The dropout rate.                                                                          0.3
====  ===========================  =========================================================================================  =====================================  =====================================================================

DRMM
####

Model Documentation
*******************

DRMM Model.

Examples:
    >>> model = DRMM()
    >>> model.params['mlp_num_layers'] = 1
    >>> model.params['mlp_num_units'] = 5
    >>> model.params['mlp_num_fan_out'] = 1
    >>> model.params['mlp_activation_func'] = 'tanh'
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================================================  ===================================  =====================================================================
  ..  Name                         Description                                                                                                                Default Value                        Default Hyper-Space
====  ===========================  =========================================================================================================================  ===================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.drmm.DRMM'>
   1  task                         Decides model output shape, loss, and metrics.
   2  out_activation_func          Activation function used in output layer.
   3  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding                    FloatTensor containing weights for the Embedding.
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  padding_idx                  If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.                                               True
  10  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                                                          128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
  11  mlp_num_layers               Number of layers of the multiple layer percetron.                                                                          3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
  12  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.                                    1                                    quantitative uniform distribution in  [4, 128), with a step size of 4
  13  mlp_activation_func          Activation function used in the multiple layer perceptron.                                                                 relu
  14  mask_value                   The value to be masked from inputs.                                                                                        0
  15  hist_bin_size                The number of bin size of the histogram.                                                                                   30
====  ===========================  =========================================================================================================================  ===================================  =====================================================================

DRMMTKS
#######

Model Documentation
*******************

DRMMTKS Model.

Examples:
    >>> model = DRMMTKS()
    >>> model.params['top_k'] = 10
    >>> model.params['mlp_num_layers'] = 1
    >>> model.params['mlp_num_units'] = 5
    >>> model.params['mlp_num_fan_out'] = 1
    >>> model.params['mlp_activation_func'] = 'tanh'
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================================================  =========================================  =====================================================================
  ..  Name                         Description                                                                                                                Default Value                              Default Hyper-Space
====  ===========================  =========================================================================================================================  =========================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.drmmtks.DRMMTKS'>
   1  task                         Decides model output shape, loss, and metrics.
   2  out_activation_func          Activation function used in output layer.
   3  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding                    FloatTensor containing weights for the Embedding.
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  padding_idx                  If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.                                               True
  10  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                                                          128                                        quantitative uniform distribution in  [8, 256), with a step size of 8
  11  mlp_num_layers               Number of layers of the multiple layer percetron.                                                                          3                                          quantitative uniform distribution in  [1, 6), with a step size of 1
  12  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.                                    1                                          quantitative uniform distribution in  [4, 128), with a step size of 4
  13  mlp_activation_func          Activation function used in the multiple layer perceptron.                                                                 relu
  14  mask_value                   The value to be masked from inputs.                                                                                        0
  15  top_k                        Size of top-k pooling layer.                                                                                               10                                         quantitative uniform distribution in  [2, 100), with a step size of 1
====  ===========================  =========================================================================================================================  =========================================  =====================================================================

ESIM
####

Model Documentation
*******************

ESIM Model.

Examples:
    >>> model = ESIM()
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================================================  ===================================  =====================
  ..  Name                  Description                                                                                                                Default Value                        Default Hyper-Space
====  ====================  =========================================================================================================================  ===================================  =====================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.esim.ESIM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  out_activation_func   Activation function used in output layer.
   3  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding             FloatTensor containing weights for the Embedding.
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  padding_idx           If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  mask_value            The value to be masked from inputs.                                                                                        0
  10  dropout               Dropout rate.                                                                                                              0.2
  11  hidden_size           Hidden size.                                                                                                               200
  12  lstm_layer            Number of LSTM layers                                                                                                      1
  13  drop_lstm             Whether dropout LSTM.                                                                                                      False
  14  concat_lstm           Whether concat intermediate outputs.                                                                                       True
  15  rnn_type              Choose rnn type, lstm or gru.                                                                                              lstm
====  ====================  =========================================================================================================================  ===================================  =====================

KNRM
####

Model Documentation
*******************

KNRM Model.

Examples:
    >>> model = KNRM()
    >>> model.params['kernel_num'] = 11
    >>> model.params['sigma'] = 0.1
    >>> model.params['exact_sigma'] = 0.001
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================================================  ===================================  ===========================================================================
  ..  Name                  Description                                                                                                                Default Value                        Default Hyper-Space
====  ====================  =========================================================================================================================  ===================================  ===========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.knrm.KNRM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  out_activation_func   Activation function used in output layer.
   3  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding             FloatTensor containing weights for the Embedding.
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  padding_idx           If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  kernel_num            The number of RBF kernels.                                                                                                 11                                   quantitative uniform distribution in  [5, 20), with a step size of 1
  10  sigma                 The `sigma` defines the kernel width.                                                                                      0.1                                  quantitative uniform distribution in  [0.01, 0.2), with a step size of 0.01
  11  exact_sigma           The `exact_sigma` denotes the `sigma` for exact match.                                                                     0.001
====  ====================  =========================================================================================================================  ===================================  ===========================================================================

ConvKNRM
########

Model Documentation
*******************

ConvKNRM Model.

Examples:
    >>> model = ConvKNRM()
    >>> model.params['filters'] = 128
    >>> model.params['conv_activation_func'] = 'tanh'
    >>> model.params['max_ngram'] = 3
    >>> model.params['use_crossmatch'] = True
    >>> model.params['kernel_num'] = 11
    >>> model.params['sigma'] = 0.1
    >>> model.params['exact_sigma'] = 0.001
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================================================  ============================================  ===========================================================================
  ..  Name                  Description                                                                                                                Default Value                                 Default Hyper-Space
====  ====================  =========================================================================================================================  ============================================  ===========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.conv_knrm.ConvKNRM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  out_activation_func   Activation function used in output layer.
   3  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding             FloatTensor containing weights for the Embedding.
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  padding_idx           If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  filters               The filter size in the convolution layer.                                                                                  128
  10  conv_activation_func  The activation function in the convolution layer.                                                                          relu
  11  max_ngram             The maximum length of n-grams for the convolution layer.                                                                   3
  12  use_crossmatch        Whether to match left n-grams and right n-grams of different lengths                                                       True
  13  kernel_num            The number of RBF kernels.                                                                                                 11                                            quantitative uniform distribution in  [5, 20), with a step size of 1
  14  sigma                 The `sigma` defines the kernel width.                                                                                      0.1                                           quantitative uniform distribution in  [0.01, 0.2), with a step size of 0.01
  15  exact_sigma           The `exact_sigma` denotes the `sigma` for exact match.                                                                     0.001
====  ====================  =========================================================================================================================  ============================================  ===========================================================================

BiMPM
#####

Model Documentation
*******************

BiMPM Model.

Reference:
- https://github.com/galsang/BIMPM-pytorch/blob/master/model/BIMPM.py

Examples:
    >>> model = BiMPM()
    >>> model.params['num_perspective'] = 4
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================================================  =====================================  =========================================================================
  ..  Name                  Description                                                                                                                Default Value                          Default Hyper-Space
====  ====================  =========================================================================================================================  =====================================  =========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.bimpm.BiMPM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  out_activation_func   Activation function used in output layer.
   3  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding             FloatTensor containing weights for the Embedding.
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  padding_idx           If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  mask_value            The value to be masked from inputs.                                                                                        0
  10  dropout               Dropout rate.                                                                                                              0.2
  11  hidden_size           Hidden size.                                                                                                               100                                    quantitative uniform distribution in  [100, 300), with a step size of 100
  12  num_perspective       num_perspective                                                                                                            20                                     quantitative uniform distribution in  [20, 100), with a step size of 20
====  ====================  =========================================================================================================================  =====================================  =========================================================================

MatchLSTM
#########

Model Documentation
*******************

MatchLSTM Model.

https://github.com/shuohangwang/mprc/blob/master/qa/rankerReader.lua.

Examples:
    >>> model = MatchLSTM()
    >>> model.params['dropout'] = 0.2
    >>> model.params['hidden_size'] = 200
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================================================  =============================================  =====================
  ..  Name                  Description                                                                                                                Default Value                                  Default Hyper-Space
====  ====================  =========================================================================================================================  =============================================  =====================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.matchlstm.MatchLSTM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  out_activation_func   Activation function used in output layer.
   3  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding             FloatTensor containing weights for the Embedding.
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  padding_idx           If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  mask_value            The value to be masked from inputs.                                                                                        0
  10  dropout               Dropout rate.                                                                                                              0.2
  11  hidden_size           Hidden size.                                                                                                               200
  12  lstm_layer            Number of LSTM layers                                                                                                      1
  13  drop_lstm             Whether dropout LSTM.                                                                                                      False
  14  concat_lstm           Whether concat intermediate outputs.                                                                                       True
  15  rnn_type              Choose rnn type, lstm or gru.                                                                                              lstm
====  ====================  =========================================================================================================================  =============================================  =====================

ArcI
####

Model Documentation
*******************

ArcI Model.

Examples:
    >>> model = ArcI()
    >>> model.params['left_filters'] = [32]
    >>> model.params['right_filters'] = [32]
    >>> model.params['left_kernel_sizes'] = [3]
    >>> model.params['right_kernel_sizes'] = [3]
    >>> model.params['left_pool_sizes'] = [2]
    >>> model.params['right_pool_sizes'] = [4]
    >>> model.params['conv_activation_func'] = 'relu'
    >>> model.params['mlp_num_layers'] = 1
    >>> model.params['mlp_num_units'] = 64
    >>> model.params['mlp_num_fan_out'] = 32
    >>> model.params['mlp_activation_func'] = 'relu'
    >>> model.params['dropout_rate'] = 0.5
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================================================  ===================================  ==========================================================================
  ..  Name                         Description                                                                                                                Default Value                        Default Hyper-Space
====  ===========================  =========================================================================================================================  ===================================  ==========================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.arci.ArcI'>
   1  task                         Decides model output shape, loss, and metrics.
   2  out_activation_func          Activation function used in output layer.
   3  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding                    FloatTensor containing weights for the Embedding.
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  padding_idx                  If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.                                               True
  10  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                                                          128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
  11  mlp_num_layers               Number of layers of the multiple layer percetron.                                                                          3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
  12  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.                                    64                                   quantitative uniform distribution in  [4, 128), with a step size of 4
  13  mlp_activation_func          Activation function used in the multiple layer perceptron.                                                                 relu
  14  left_length                  Length of left input.                                                                                                      10
  15  right_length                 Length of right input.                                                                                                     100
  16  conv_activation_func         The activation function in the convolution layer.                                                                          relu
  17  left_filters                 The filter size of each convolution blocks for the left input.                                                             [32]
  18  left_kernel_sizes            The kernel size of each convolution blocks for the left input.                                                             [3]
  19  left_pool_sizes              The pooling size of each convolution blocks for the left input.                                                            [2]
  20  right_filters                The filter size of each convolution blocks for the right input.                                                            [32]
  21  right_kernel_sizes           The kernel size of each convolution blocks for the right input.                                                            [3]
  22  right_pool_sizes             The pooling size of each convolution blocks for the right input.                                                           [2]
  23  dropout_rate                 The dropout rate.                                                                                                          0.0                                  quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ===========================  =========================================================================================================================  ===================================  ==========================================================================

ArcII
#####

Model Documentation
*******************

ArcII Model.

Examples:
    >>> model = ArcII()
    >>> model.params['embedding_output_dim'] = 300
    >>> model.params['kernel_1d_count'] = 32
    >>> model.params['kernel_1d_size'] = 3
    >>> model.params['kernel_2d_count'] = [16, 32]
    >>> model.params['kernel_2d_size'] = [[3, 3], [3, 3]]
    >>> model.params['pool_2d_size'] = [[2, 2], [2, 2]]
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================================================  =====================================  ==========================================================================
  ..  Name                  Description                                                                                                                Default Value                          Default Hyper-Space
====  ====================  =========================================================================================================================  =====================================  ==========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.arcii.ArcII'>
   1  task                  Decides model output shape, loss, and metrics.
   2  out_activation_func   Activation function used in output layer.
   3  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding             FloatTensor containing weights for the Embedding.
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  padding_idx           If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  left_length           Length of left input.                                                                                                      10
  10  right_length          Length of right input.                                                                                                     100
  11  kernel_1d_count       Kernel count of 1D convolution layer.                                                                                      32
  12  kernel_1d_size        Kernel size of 1D convolution layer.                                                                                       3
  13  kernel_2d_count       Kernel count of 2D convolution layer ineach block                                                                          [32]
  14  kernel_2d_size        Kernel size of 2D convolution layer in each block.                                                                         [(3, 3)]
  15  activation            Activation function.                                                                                                       relu
  16  pool_2d_size          Size of pooling layer in each block.                                                                                       [(2, 2)]
  17  dropout_rate          The dropout rate.                                                                                                          0.0                                    quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ====================  =========================================================================================================================  =====================================  ==========================================================================

Bert
####

Model Documentation
*******************

Bert Model.

Model Hyper Parameters
**********************

====  ===================  =========================================================================================  ===================================  ==========================================================================
  ..  Name                 Description                                                                                Default Value                        Default Hyper-Space
====  ===================  =========================================================================================  ===================================  ==========================================================================
   0  model_class          Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.bert.Bert'>
   1  task                 Decides model output shape, loss, and metrics.
   2  out_activation_func  Activation function used in output layer.
   3  mode                 Pretrained Bert model.                                                                     bert-base-uncased
   4  dropout_rate         The dropout rate.                                                                          0.0                                  quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ===================  =========================================================================================  ===================================  ==========================================================================

MVLSTM
######

Model Documentation
*******************

MVLSTM Model.

Examples:
    >>> model = MVLSTM()
    >>> model.params['hidden_size'] = 32
    >>> model.params['top_k'] = 50
    >>> model.params['mlp_num_layers'] = 2
    >>> model.params['mlp_num_units'] = 20
    >>> model.params['mlp_num_fan_out'] = 10
    >>> model.params['mlp_activation_func'] = 'relu'
    >>> model.params['dropout_rate'] = 0.0
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================================================  =======================================  ==========================================================================
  ..  Name                         Description                                                                                                                Default Value                            Default Hyper-Space
====  ===========================  =========================================================================================================================  =======================================  ==========================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.mvlstm.MVLSTM'>
   1  task                         Decides model output shape, loss, and metrics.
   2  out_activation_func          Activation function used in output layer.
   3  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding                    FloatTensor containing weights for the Embedding.
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  padding_idx                  If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.                                               True
  10  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                                                          128                                      quantitative uniform distribution in  [8, 256), with a step size of 8
  11  mlp_num_layers               Number of layers of the multiple layer percetron.                                                                          3                                        quantitative uniform distribution in  [1, 6), with a step size of 1
  12  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.                                    64                                       quantitative uniform distribution in  [4, 128), with a step size of 4
  13  mlp_activation_func          Activation function used in the multiple layer perceptron.                                                                 relu
  14  hidden_size                  Integer, the hidden size in the bi-directional LSTM layer.                                                                 32
  15  num_layers                   Integer, number of recurrent layers.                                                                                       1
  16  top_k                        Size of top-k pooling layer.                                                                                               10                                       quantitative uniform distribution in  [2, 100), with a step size of 1
  17  dropout_rate                 Float, the dropout rate.                                                                                                   0.0                                      quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ===========================  =========================================================================================================================  =======================================  ==========================================================================

MatchPyramid
############

Model Documentation
*******************

MatchPyramid Model.

Examples:
    >>> model = MatchPyramid()
    >>> model.params['embedding_output_dim'] = 300
    >>> model.params['kernel_count'] = [16, 32]
    >>> model.params['kernel_size'] = [[3, 3], [3, 3]]
    >>> model.params['dpool_size'] = [3, 10]
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================================================  ====================================================  ==========================================================================
  ..  Name                  Description                                                                                                                Default Value                                         Default Hyper-Space
====  ====================  =========================================================================================================================  ====================================================  ==========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.match_pyramid.MatchPyramid'>
   1  task                  Decides model output shape, loss, and metrics.
   2  out_activation_func   Activation function used in output layer.
   3  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding             FloatTensor containing weights for the Embedding.
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  padding_idx           If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  kernel_count          The kernel count of the 2D convolution of each block.                                                                      [32]
  10  kernel_size           The kernel size of the 2D convolution of each block.                                                                       [[3, 3]]
  11  activation            The activation function.                                                                                                   relu
  12  dpool_size            The max-pooling size of each block.                                                                                        [3, 10]
  13  dropout_rate          The dropout rate.                                                                                                          0.0                                                   quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ====================  =========================================================================================================================  ====================================================  ==========================================================================

aNMM
####

Model Documentation
*******************

aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model.

Examples:
    >>> model = aNMM()
    >>> model.params['embedding_output_dim'] = 300
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================================================  ===================================  ==========================================================================
  ..  Name                  Description                                                                                                                Default Value                        Default Hyper-Space
====  ====================  =========================================================================================================================  ===================================  ==========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.anmm.aNMM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  out_activation_func   Activation function used in output layer.
   3  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding             FloatTensor containing weights for the Embedding.
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  padding_idx           If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  mask_value            The value to be masked from inputs.                                                                                        0
  10  num_bins              Integer, number of bins.                                                                                                   200
  11  hidden_sizes          Number of hidden size for each hidden layer                                                                                [100]
  12  activation            The activation function.                                                                                                   relu
  13  dropout_rate          The dropout rate.                                                                                                          0.0                                  quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ====================  =========================================================================================================================  ===================================  ==========================================================================

HBMP
####

Model Documentation
*******************

HBMP model.

Examples:
    >>> model = HBMP()
    >>> model.params['embedding_input_dim'] = 200
    >>> model.params['embedding_output_dim'] = 100
    >>> model.params['mlp_num_layers'] = 1
    >>> model.params['mlp_num_units'] = 10
    >>> model.params['mlp_num_fan_out'] = 10
    >>> model.params['mlp_activation_func'] = nn.LeakyReLU(0.1)
    >>> model.params['lstm_hidden_size'] = 5
    >>> model.params['lstm_num'] = 3
    >>> model.params['num_layers'] = 3
    >>> model.params['dropout_rate'] = 0.1
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================================================  ===================================  ==========================================================================
  ..  Name                         Description                                                                                                                Default Value                        Default Hyper-Space
====  ===========================  =========================================================================================================================  ===================================  ==========================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.hbmp.HBMP'>
   1  task                         Decides model output shape, loss, and metrics.
   2  out_activation_func          Activation function used in output layer.
   3  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding                    FloatTensor containing weights for the Embedding.
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  padding_idx                  If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.                                               True
  10  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                                                          128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
  11  mlp_num_layers               Number of layers of the multiple layer percetron.                                                                          3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
  12  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.                                    64                                   quantitative uniform distribution in  [4, 128), with a step size of 4
  13  mlp_activation_func          Activation function used in the multiple layer perceptron.                                                                 relu
  14  lstm_hidden_size             Integer, the hidden size of the bi-directional LSTM layer.                                                                 5
  15  lstm_num                     Integer, number of LSTM units                                                                                              3
  16  num_layers                   Integer, number of LSTM layers.                                                                                            1
  17  dropout_rate                 The dropout rate.                                                                                                          0.0                                  quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ===========================  =========================================================================================================================  ===================================  ==========================================================================

DUET
####

Model Documentation
*******************

Duet Model.

Examples:
    >>> model = DUET()
    >>> model.params['left_length'] = 10
    >>> model.params['right_length'] = 40
    >>> model.params['lm_filters'] = 300
    >>> model.params['mlp_num_layers'] = 2
    >>> model.params['mlp_num_units'] = 300
    >>> model.params['mlp_num_fan_out'] = 300
    >>> model.params['mlp_activation_func'] = 'relu'
    >>> model.params['vocab_size'] = 2000
    >>> model.params['dm_filters'] = 300
    >>> model.params['dm_conv_activation_func'] = 'relu'
    >>> model.params['dm_kernel_size'] = 3
    >>> model.params['dm_right_pool_size'] = 8
    >>> model.params['dropout_rate'] = 0.5
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================  ===================================  ==========================================================================
  ..  Name                         Description                                                                                Default Value                        Default Hyper-Space
====  ===========================  =========================================================================================  ===================================  ==========================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.duet.DUET'>
   1  task                         Decides model output shape, loss, and metrics.
   2  out_activation_func          Activation function used in output layer.
   3  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   4  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
   5  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
   6  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                   quantitative uniform distribution in  [4, 128), with a step size of 4
   7  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
   8  mask_value                   The value to be masked from inputs.                                                        0
   9  left_length                  Length of left input.                                                                      10
  10  right_length                 Length of right input.                                                                     40
  11  lm_filters                   Filter size of 1D convolution layer in the local model.                                    300
  12  vocab_size                   Vocabulary size of the tri-letters used in the distributed model.                          419
  13  dm_filters                   Filter size of 1D convolution layer in the distributed model.                              300
  14  dm_kernel_size               Kernel size of 1D convolution layer in the distributed model.                              3
  15  dm_conv_activation_func      Activation functions of the convolution layer in the distributed model.                    relu
  16  dm_right_pool_size           Kernel size of 1D convolution layer in the distributed model.                              8
  17  dropout_rate                 The dropout rate.                                                                          0.5                                  quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.02
====  ===========================  =========================================================================================  ===================================  ==========================================================================

DIIN
####

Model Documentation
*******************

DIIN model.

Examples:
    >>> model = DIIN()
    >>> model.params['embedding_input_dim'] = 10000
    >>> model.params['embedding_output_dim'] = 300
    >>> model.params['mask_value'] = 0
    >>> model.params['char_embedding_input_dim'] = 100
    >>> model.params['char_embedding_output_dim'] = 8
    >>> model.params['char_conv_filters'] = 100
    >>> model.params['char_conv_kernel_size'] = 5
    >>> model.params['first_scale_down_ratio'] = 0.3
    >>> model.params['nb_dense_blocks'] = 3
    >>> model.params['layers_per_dense_block'] = 8
    >>> model.params['growth_rate'] = 20
    >>> model.params['transition_scale_down_ratio'] = 0.5
    >>> model.params['conv_kernel_size'] = (3, 3)
    >>> model.params['pool_kernel_size'] = (2, 2)
    >>> model.params['dropout_rate'] = 0.2
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================================================  ===================================  ==========================================================================
  ..  Name                         Description                                                                                                                Default Value                        Default Hyper-Space
====  ===========================  =========================================================================================================================  ===================================  ==========================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.diin.DIIN'>
   1  task                         Decides model output shape, loss, and metrics.
   2  out_activation_func          Activation function used in output layer.
   3  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding                    FloatTensor containing weights for the Embedding.
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  padding_idx                  If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  mask_value                   The value to be masked from inputs.                                                                                        0
  10  char_embedding_input_dim     The input dimension of character embedding layer.                                                                          100
  11  char_embedding_output_dim    The output dimension of character embedding layer.                                                                         8
  12  char_conv_filters            The filter size of character convolution layer.                                                                            100
  13  char_conv_kernel_size        The kernel size of character convolution layer.                                                                            5
  14  first_scale_down_ratio       The channel scale down ratio of the convolution layer before densenet.                                                     0.3
  15  nb_dense_blocks              The number of blocks in densenet.                                                                                          3
  16  layers_per_dense_block       The number of convolution layers in dense block.                                                                           8
  17  growth_rate                  The filter size of each convolution layer in dense block.                                                                  20
  18  transition_scale_down_ratio  The channel scale down ratio of the convolution layer in transition block.                                                 0.5
  19  conv_kernel_size             The kernel size of convolution layer in dense block.                                                                       (3, 3)
  20  pool_kernel_size             The kernel size of pooling layer in transition block.                                                                      (2, 2)
  21  dropout_rate                 The dropout rate.                                                                                                          0.0                                  quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ===========================  =========================================================================================================================  ===================================  ==========================================================================

MatchSRNN
#########

Model Documentation
*******************

Match-SRNN Model.

Examples:
    >>> model = MatchSRNN()
    >>> model.params['channels'] = 4
    >>> model.params['units'] = 10
    >>> model.params['dropout'] = 0.2
    >>> model.params['direction'] = 'lt'
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================================================  ==============================================  ==========================================================================
  ..  Name                  Description                                                                                                                Default Value                                   Default Hyper-Space
====  ====================  =========================================================================================================================  ==============================================  ==========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.                                  <class 'matchzoo.models.match_srnn.MatchSRNN'>
   1  task                  Decides model output shape, loss, and metrics.
   2  out_activation_func   Activation function used in output layer.
   3  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                                                      True
   4  embedding             FloatTensor containing weights for the Embedding.
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  padding_idx           If given, pads the output with the embedding vector atpadding_idx (initialized to zeros) whenever it encountersthe index.  0
   8  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.                                         False
   9  channels              Number of word interaction tensor channels                                                                                 4
  10  units                 Number of SpatialGRU units                                                                                                 10
  11  direction             Direction of SpatialGRU scanning                                                                                           lt
  12  dropout               The dropout rate.                                                                                                          0.2                                             quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ====================  =========================================================================================================================  ==============================================  ==========================================================================

