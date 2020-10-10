cnn <- torch::nn_module(
  "Net",
  initialize = function(n_classes) {
    self$conv1 <- torch::nn_conv2d(1, 32, 3, 1)
    self$conv2 <- torch::nn_conv2d(32, 64, 3, 1)
    self$fc1 <- torch::nn_linear(9216, 128)
    self$fc2 <- torch::nn_linear(128, n_classes)
  },
  forward = function(x) {
    x <- self$conv1(x)
    x <- torch::nnf_relu(x)
    x <- self$conv2(x)
    x <- torch::nnf_relu(x)
    x <- torch::nnf_max_pool2d(x, 2)
    x <- torch::torch_flatten(x, start_dim = 2)
    x <- self$fc1(x)
    x <- torch::nnf_relu(x)
    self$fc2(x)
  }
)
