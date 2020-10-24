size_after_convs <- function(in_size, kernel_size, n_conv) {
  new_size <- in_size - (kernel_size - 1) - 1 + 1
  
  if (n_conv == 1)
    new_size
  else
    size_after_convs(new_size, kernel_size, n_conv -1)
}

size_after_max_pool <- function(in_size, pool_size) {
  in_size/pool_size
}

size_for_flatten <- function(in_size) {
  x <- size_after_convs(in_size, kernel_size = 3, n_conv = 2)
  x <- size_after_max_pool(x, 2)
  (x^2)*64
}

cnn <- torch::nn_module(
  "Net",
  initialize = function(n_classes, input_dim) {
    self$conv1 <- torch::nn_conv2d(in_channels = input_dim[1], out_channels = 32, kernel_size = 3, stride = 1)
    self$conv2 <- torch::nn_conv2d(32, 64, 3, 1)
    self$fc1 <- torch::nn_linear(size_for_flatten(input_dim[2]), 128)
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
