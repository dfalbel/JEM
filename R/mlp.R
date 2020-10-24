mlp <- torch::nn_module(
  initialize = function(n_classes, input_dim) {
    self$l1 = torch::nn_linear(prod(input_dim), 1000)
    self$l2 = torch::nn_linear(1000, 1000)
    self$l3 = torch::nn_linear(1000, 10)
  },
  forward = function(x) {
    x %>%
      torch::torch_flatten(start_dim = 2) %>%
      self$l1() %>%
      torch::nnf_relu() %>%
      self$l2() %>%
      torch::nnf_relu() %>%
      self$l3()
  }
)
