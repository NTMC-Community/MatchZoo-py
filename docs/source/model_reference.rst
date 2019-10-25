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

====  ===========================  =========================================================================================  ======================================================  ======================================================================
  ..  Name                         Description                                                                                Default Value                                           Default Hyper-Space
====  ===========================  =========================================================================================  ======================================================  ======================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.dense_baseline.DenseBaseline'>
   1  task                         Decides model output shape, loss, and metrics.
   2  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding                    FloatTensor containing weights for the Embedding.
   4  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim         Should be set manually.
   6  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   8  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          256                                                     quantitative uniform distribution in  [16, 512), with a step size of 1
   9  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                                       quantitative uniform distribution in  [1, 5), with a step size of 1
  10  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                                      quantitative uniform distribution in  [4, 128), with a step size of 4
  11  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
====  ===========================  =========================================================================================  ======================================================  ======================================================================

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
   2  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   3  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
   4  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
   5  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                   quantitative uniform distribution in  [4, 128), with a step size of 4
   6  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
   7  vocab_size                   Size of vocabulary.                                                                        419
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
   2  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   3  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                    quantitative uniform distribution in  [8, 256), with a step size of 8
   4  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                      quantitative uniform distribution in  [1, 6), with a step size of 1
   5  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                     quantitative uniform distribution in  [4, 128), with a step size of 4
   6  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
   7  vocab_size                   Size of vocabulary.                                                                        419
   8  filters                      Number of filters in the 1D convolution layer.                                             3
   9  kernel_size                  Number of kernel size in the 1D convolution layer.                                         3
  10  conv_activation_func         Activation function in the convolution layer.                                              relu
  11  dropout_rate                 The dropout rate.                                                                          0.3
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

====  ===========================  =========================================================================================  ===================================  =====================================================================
  ..  Name                         Description                                                                                Default Value                        Default Hyper-Space
====  ===========================  =========================================================================================  ===================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.drmm.DRMM'>
   1  task                         Decides model output shape, loss, and metrics.
   2  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding                    FloatTensor containing weights for the Embedding.
   4  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim         Should be set manually.
   6  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   8  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
   9  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
  10  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    1                                    quantitative uniform distribution in  [4, 128), with a step size of 4
  11  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
  12  mask_value                   The value to be masked from inputs.                                                        0
  13  hist_bin_size                The number of bin size of the histogram.                                                   30
====  ===========================  =========================================================================================  ===================================  =====================================================================

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

====  ===========================  =========================================================================================  =========================================  =====================================================================
  ..  Name                         Description                                                                                Default Value                              Default Hyper-Space
====  ===========================  =========================================================================================  =========================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.drmmtks.DRMMTKS'>
   1  task                         Decides model output shape, loss, and metrics.
   2  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding                    FloatTensor containing weights for the Embedding.
   4  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim         Should be set manually.
   6  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   8  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                        quantitative uniform distribution in  [8, 256), with a step size of 8
   9  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                          quantitative uniform distribution in  [1, 6), with a step size of 1
  10  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    1                                          quantitative uniform distribution in  [4, 128), with a step size of 4
  11  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
  12  mask_value                   The value to be masked from inputs.                                                        0
  13  top_k                        Size of top-k pooling layer.                                                               10                                         quantitative uniform distribution in  [2, 100), with a step size of 1
====  ===========================  =========================================================================================  =========================================  =====================================================================

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

====  ====================  =========================================================================================  ===================================  =====================
  ..  Name                  Description                                                                                Default Value                        Default Hyper-Space
====  ====================  =========================================================================================  ===================================  =====================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.esim.ESIM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding             FloatTensor containing weights for the Embedding.
   4  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim  Should be set manually.
   6  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  mask_value            The value to be masked from inputs.                                                        0
   8  dropout               Dropout rate.                                                                              0.2
   9  hidden_size           Hidden size.                                                                               200
  10  lstm_layer            Number of LSTM layers                                                                      1
  11  drop_lstm             Whether dropout LSTM.                                                                      False
  12  concat_lstm           Whether concat intermediate outputs.                                                       True
  13  rnn_type              Choose rnn type, lstm or gru.                                                              lstm
====  ====================  =========================================================================================  ===================================  =====================

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

====  ====================  =========================================================================================  ===================================  ===========================================================================
  ..  Name                  Description                                                                                Default Value                        Default Hyper-Space
====  ====================  =========================================================================================  ===================================  ===========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.knrm.KNRM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding             FloatTensor containing weights for the Embedding.
   4  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim  Should be set manually.
   6  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  kernel_num            The number of RBF kernels.                                                                 11                                   quantitative uniform distribution in  [5, 20), with a step size of 1
   8  sigma                 The `sigma` defines the kernel width.                                                      0.1                                  quantitative uniform distribution in  [0.01, 0.2), with a step size of 0.01
   9  exact_sigma           The `exact_sigma` denotes the `sigma` for exact match.                                     0.001
====  ====================  =========================================================================================  ===================================  ===========================================================================

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

====  ====================  =========================================================================================  ============================================  ===========================================================================
  ..  Name                  Description                                                                                Default Value                                 Default Hyper-Space
====  ====================  =========================================================================================  ============================================  ===========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.conv_knrm.ConvKNRM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding             FloatTensor containing weights for the Embedding.
   4  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim  Should be set manually.
   6  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  filters               The filter size in the convolution layer.                                                  128
   8  conv_activation_func  The activation function in the convolution layer.                                          relu
   9  max_ngram             The maximum length of n-grams for the convolution layer.                                   3
  10  use_crossmatch        Whether to match left n-grams and right n-grams of different lengths                       True
  11  kernel_num            The number of RBF kernels.                                                                 11                                            quantitative uniform distribution in  [5, 20), with a step size of 1
  12  sigma                 The `sigma` defines the kernel width.                                                      0.1                                           quantitative uniform distribution in  [0.01, 0.2), with a step size of 0.01
  13  exact_sigma           The `exact_sigma` denotes the `sigma` for exact match.                                     0.001
====  ====================  =========================================================================================  ============================================  ===========================================================================

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

====  ====================  =========================================================================================  =====================================  =========================================================================
  ..  Name                  Description                                                                                Default Value                          Default Hyper-Space
====  ====================  =========================================================================================  =====================================  =========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.bimpm.BiMPM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding             FloatTensor containing weights for the Embedding.
   4  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim  Should be set manually.
   6  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  mask_value            The value to be masked from inputs.                                                        0
   8  dropout               Dropout rate.                                                                              0.2
   9  hidden_size           Hidden size.                                                                               100                                    quantitative uniform distribution in  [100, 300), with a step size of 100
  10  num_perspective       num_perspective                                                                            20                                     quantitative uniform distribution in  [20, 100), with a step size of 20
====  ====================  =========================================================================================  =====================================  =========================================================================

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

====  ====================  =========================================================================================  =============================================  =====================
  ..  Name                  Description                                                                                Default Value                                  Default Hyper-Space
====  ====================  =========================================================================================  =============================================  =====================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.matchlstm.MatchLSTM'>
   1  task                  Decides model output shape, loss, and metrics.
   2  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding             FloatTensor containing weights for the Embedding.
   4  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim  Should be set manually.
   6  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  mask_value            The value to be masked from inputs.                                                        0
   8  dropout               Dropout rate.                                                                              0.2
   9  hidden_size           Hidden size.                                                                               200
  10  lstm_layer            Number of LSTM layers                                                                      1
  11  drop_lstm             Whether dropout LSTM.                                                                      False
  12  concat_lstm           Whether concat intermediate outputs.                                                       True
  13  rnn_type              Choose rnn type, lstm or gru.                                                              lstm
====  ====================  =========================================================================================  =============================================  =====================

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

====  ===========================  =========================================================================================  ===================================  ==========================================================================
  ..  Name                         Description                                                                                Default Value                        Default Hyper-Space
====  ===========================  =========================================================================================  ===================================  ==========================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.arci.ArcI'>
   1  task                         Decides model output shape, loss, and metrics.
   2  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding                    FloatTensor containing weights for the Embedding.
   4  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim         Should be set manually.
   6  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   8  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
   9  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
  10  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                   quantitative uniform distribution in  [4, 128), with a step size of 4
  11  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
  12  left_length                  Length of left input.                                                                      10
  13  right_length                 Length of right input.                                                                     100
  14  conv_activation_func         The activation function in the convolution layer.                                          relu
  15  left_filters                 The filter size of each convolution blocks for the left input.                             [32]
  16  left_kernel_sizes            The kernel size of each convolution blocks for the left input.                             [3]
  17  left_pool_sizes              The pooling size of each convolution blocks for the left input.                            [2]
  18  right_filters                The filter size of each convolution blocks for the right input.                            [32]
  19  right_kernel_sizes           The kernel size of each convolution blocks for the right input.                            [3]
  20  right_pool_sizes             The pooling size of each convolution blocks for the right input.                           [2]
  21  dropout_rate                 The dropout rate.                                                                          0.0                                  quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ===========================  =========================================================================================  ===================================  ==========================================================================

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

====  ====================  =========================================================================================  =====================================  ==========================================================================
  ..  Name                  Description                                                                                Default Value                          Default Hyper-Space
====  ====================  =========================================================================================  =====================================  ==========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.arcii.ArcII'>
   1  task                  Decides model output shape, loss, and metrics.
   2  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding             FloatTensor containing weights for the Embedding.
   4  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim  Should be set manually.
   6  embedding_freeze      `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  left_length           Length of left input.                                                                      10
   8  right_length          Length of right input.                                                                     100
   9  kernel_1d_count       Kernel count of 1D convolution layer.                                                      32
  10  kernel_1d_size        Kernel size of 1D convolution layer.                                                       3
  11  kernel_2d_count       Kernel count of 2D convolution layer ineach block                                          [32]
  12  kernel_2d_size        Kernel size of 2D convolution layer in each block.                                         [(3, 3)]
  13  activation            Activation function.                                                                       relu
  14  pool_2d_size          Size of pooling layer in each block.                                                       [(2, 2)]
  15  dropout_rate          The dropout rate.                                                                          0.0                                    quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ====================  =========================================================================================  =====================================  ==========================================================================

Bert
####

Model Documentation
*******************

Bert Model.

Model Hyper Parameters
**********************

====  ============  =========================================================================================  ===================================  ==========================================================================
  ..  Name          Description                                                                                Default Value                        Default Hyper-Space
====  ============  =========================================================================================  ===================================  ==========================================================================
   0  model_class   Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.bert.Bert'>
   1  task          Decides model output shape, loss, and metrics.
   2  mode          Pretrained Bert model.                                                                     bert-base-uncased
   3  dropout_rate  The dropout rate.                                                                          0.0                                  quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ============  =========================================================================================  ===================================  ==========================================================================

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

====  ===========================  =========================================================================================  =======================================  ==========================================================================
  ..  Name                         Description                                                                                Default Value                            Default Hyper-Space
====  ===========================  =========================================================================================  =======================================  ==========================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.mvlstm.MVLSTM'>
   1  task                         Decides model output shape, loss, and metrics.
   2  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                      True
   3  embedding                    FloatTensor containing weights for the Embedding.
   4  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   5  embedding_output_dim         Should be set manually.
   6  embedding_freeze             `True` to freeze embedding layer training, `False` to enable embedding parameters.         False
   7  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   8  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                      quantitative uniform distribution in  [8, 256), with a step size of 8
   9  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                        quantitative uniform distribution in  [1, 6), with a step size of 1
  10  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                       quantitative uniform distribution in  [4, 128), with a step size of 4
  11  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
  12  hidden_size                  Integer, the hidden size in the bi-directional LSTM layer.                                 32
  13  num_layers                   Integer, number of recurrent layers.                                                       1
  14  top_k                        Size of top-k pooling layer.                                                               10                                       quantitative uniform distribution in  [2, 100), with a step size of 1
  15  dropout_rate                 Float, the dropout rate.                                                                   0.0                                      quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ===========================  =========================================================================================  =======================================  ==========================================================================

