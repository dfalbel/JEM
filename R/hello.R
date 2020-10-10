bar <- function(length) {
  progress::progress_bar$new(
    format = "[:bar] :eta Loss: :loss",
    total = length
  )
}

logsumexp <- function(x) {
  torch::torch_logsumexp(x, dim = 2)$view(c(x$shape[1], 1))
}

sgld_sample <- function(model, buffer) {

  rho <- config::get("rho")
  eta <- config::get("eta")
  alpha <- config::get("alpha")
  sigma <- config::get("sigma")
  batch_size <- config::get("batch_size")
  device <- config::get("device")

  n <- rbinom(1, batch_size, prob = 1 - rho)
  buffer_batch <- buffer$get_batch(n)

  size <- c(batch_size - n, buffer_batch$shape[-1])
  random_batch <- torch::torch_empty(size = size)$uniform_(-1, 1)$to(device = device)

  x <- torch::torch_cat(list(buffer_batch, random_batch))
  x <- x$to(device = device)
  x$requires_grad_(TRUE)

  for (i in seq_len(eta)) {
    y <- logsumexp(model(x))
    jacobian <- torch::autograd_grad(
      outputs = y,
      inputs = x,
      grad_outputs = torch::torch_ones_like(y),
      retain_graph = TRUE,
      create_graph = TRUE,
      allow_unused = TRUE
    )
    noise <- torch::torch_randn(size = x$shape, device = device) * sigma
    x <- x + jacobian[[1]] * alpha + noise
  }
  x <- x$detach()
  buffer$add(x)

  x
}

train <- function(model, buffer, train_dl, optimizer, loss_fn) {

  device <- config::get("device")

  for (epoch in seq_len(config::get("n_epochs"))) {

    losses <- c()
    pb <- bar(length = length(train_dl))

    for (batch in torch::enumerate(train_dl)) {

      data <- batch[[1]]$to(device = device)
      targets <- batch[[2]]$to(device = device)

      # classifier loss
      logits <- model(data)
      loss_clf <- loss_fn(logits, targets)

      # EBM loss
      data_sample <- sgld_sample(model, buffer)
      logsumexp_sample <- logsumexp(model(data_sample))
      logsumexp_data <- logsumexp(model(data))

      loss_gen <- -(logsumexp_data - logsumexp_sample)$mean()

      L <- loss_clf + loss_gen

      optimizer$zero_grad()
      L$backward()
      optimizer$step()

      losses <- c(losses, L$item())
      pb$tick(tokens = list(loss = mean(losses)))

    }

    cat(sprintf("[Epoch %d] Loss: %3f\n", epoch, mean(losses)))
  }

}

main <- function() {

  model <- cnn(n_classes = 10)
  model <- mlp(n_classes = 10)
  model$to(device = config::get("device"))

  mnist_dataset <- torchvision::mnist_dataset(
    root = ".",
    download = TRUE,
    transform = function(x) {
      torchvision::transform_to_tensor(x)$mul(2)$sub(1)
    }
  )

  buffer <- replay_buffer(buffer_size = config::get("buffer_size"))
  init <- torch::torch_empty(
    size = c(100, mnist_dataset[1][[1]]$shape),
    device = config::get("device")
  )
  buffer$add(init)

  train_dl <- torch::dataloader(
    dataset = mnist_dataset,
    batch_size = config::get("batch_size"),
    shuffle = TRUE
  )

  optimizer <- torch::optim_adam(
    params = model$parameters,
    lr = config::get("lr")
  )

  scheduler <- torch::lr_step(
    optimizer = optimizer,
    step_size = 1,
    gamma = config::get("decay_rate")
  )

  loss_fn <- torch::nn_cross_entropy_loss()
  train(model, buffer, train_dl, optimizer, loss_fn)

}
